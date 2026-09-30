package dtln

import (
	"container/heap"
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// errSchedulerClosed reports submission after scheduler shutdown
var errSchedulerClosed = errors.New("DTLN scheduler is closed")

// errStreamProcessing reports concurrent operations for one stream
var errStreamProcessing = errors.New("DTLN stream already has a chunk in progress")

// errStreamClosed reports operations after stream shutdown
var errStreamClosed = errors.New("DTLN scheduled stream is closed")

// ScheduleMetrics reports time spent waiting for and executing DTLN inference
type ScheduleMetrics struct {
	// QueueDuration measures time from submission until worker execution
	QueueDuration time.Duration
	// ProcessingDuration measures time spent executing the processor operation
	ProcessingDuration time.Duration
	// DeadlineMissed reports completion after the effective deadline
	DeadlineMissed bool
}

// Scheduler prioritizes DTLN stream chunks by deadline while limiting concurrent inference
// Each stream retains its own Processor because recurrent model state is stream specific
type Scheduler struct {
	requests      chan *scheduledRequest
	work          chan *scheduledRequest
	completed     chan scheduledCompletion
	closeRequests chan chan struct{}
	done          chan struct{}
	closed        atomic.Bool
}

// ScheduledStream associates one stateful DTLN processor with a scheduler
type ScheduledStream struct {
	scheduler *Scheduler
	processor *Processor
	request   scheduledRequest
	result    chan scheduledResult
	inFlight  atomic.Bool
	stateMu   sync.Mutex
	idle      chan struct{}
	closed    bool
	closeOnce sync.Once
	closeDone chan struct{}
	closeErr  error
}

// scheduledOperation identifies work executed by a scheduler worker
type scheduledOperation uint8

const (
	// processOperation consumes one audio chunk
	processOperation scheduledOperation = iota
	// flushOperation emits one final partial hop
	flushOperation
)

// scheduledRequest carries one stream operation through the scheduler
type scheduledRequest struct {
	context   context.Context
	processor *Processor
	deadline  time.Time
	enqueued  time.Time
	sequence  uint64
	operation scheduledOperation
	dst       []float32
	samples   []float32
	result    chan scheduledResult
}

// scheduledResult contains operation output and timing information
type scheduledResult struct {
	output  []float32
	metrics ScheduleMetrics
	err     error
}

// scheduledCompletion associates a worker result with its request
type scheduledCompletion struct {
	request *scheduledRequest
	result  scheduledResult
}

// scheduledQueue orders requests by deadline and submission sequence
type scheduledQueue []*scheduledRequest

// Len returns the number of queued requests
func (q scheduledQueue) Len() int {
	return len(q)
}

// Less reports whether one request has higher scheduling priority
func (q scheduledQueue) Less(i, j int) bool {
	left := q[i]
	right := q[j]

	if left.deadline.IsZero() != right.deadline.IsZero() {
		return !left.deadline.IsZero()
	}

	if !left.deadline.Equal(right.deadline) {
		return left.deadline.Before(right.deadline)
	}

	return left.sequence < right.sequence
}

// Swap exchanges two queued requests
func (q scheduledQueue) Swap(i, j int) {
	q[i], q[j] = q[j], q[i]
}

// Push appends one request to the queue
func (q *scheduledQueue) Push(value any) {
	*q = append(*q, value.(*scheduledRequest))
}

// Pop removes the final request selected by the heap implementation
func (q *scheduledQueue) Pop() any {
	old := *q
	last := len(old) - 1
	value := old[last]
	old[last] = nil
	*q = old[:last]

	return value
}

// NewScheduler creates a scheduler with the requested maximum concurrent inference calls
func NewScheduler(workerCount int) (*Scheduler, error) {
	if workerCount <= 0 {
		return nil, fmt.Errorf("DTLN scheduler worker count must be greater than zero")
	}

	scheduler := &Scheduler{
		requests:      make(chan *scheduledRequest),
		work:          make(chan *scheduledRequest, workerCount),
		completed:     make(chan scheduledCompletion, workerCount),
		closeRequests: make(chan chan struct{}),
		done:          make(chan struct{}),
	}

	for range workerCount {
		go scheduler.worker()
	}
	go scheduler.dispatch(workerCount)

	return scheduler, nil
}

// NewStream registers a stateful processor for deadline aware scheduling
func (s *Scheduler) NewStream(processor *Processor) (*ScheduledStream, error) {
	if processor == nil {
		return nil, fmt.Errorf("DTLN scheduler processor is nil")
	}

	if s.closed.Load() {
		return nil, errSchedulerClosed
	}

	idle := make(chan struct{}, 1)
	idle <- struct{}{}

	return &ScheduledStream{
		scheduler: s,
		processor: processor,
		result:    make(chan scheduledResult, 1),
		idle:      idle,
		closeDone: make(chan struct{}),
	}, nil
}

// Process schedules one stream chunk and blocks until processing completes
// The caller must not reuse samples or dst until Process returns
func (s *ScheduledStream) Process(
	ctx context.Context, deadline time.Time, dst, samples []float32,
) ([]float32, ScheduleMetrics, error) {
	return s.schedule(ctx, deadline, processOperation, dst, samples)
}

// Flush schedules the final partial hop and appends it to dst without allocating
func (s *ScheduledStream) Flush(
	ctx context.Context, deadline time.Time, dst []float32,
) ([]float32, ScheduleMetrics, error) {
	return s.schedule(ctx, deadline, flushOperation, dst, nil)
}

// schedule submits one process or flush operation and waits for its result
func (s *ScheduledStream) schedule(
	ctx context.Context, deadline time.Time, operation scheduledOperation, dst, samples []float32,
) ([]float32, ScheduleMetrics, error) {
	if err := s.beginProcess(ctx); err != nil {
		return nil, ScheduleMetrics{}, err
	}
	defer s.finishProcess()

	request := s.prepareRequest(ctx, deadline, operation, dst, samples)
	if err := s.submit(ctx, request); err != nil {
		return nil, ScheduleMetrics{}, err
	}

	return s.awaitResult()
}

// beginProcess reserves the stream for one operation
func (s *ScheduledStream) beginProcess(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("DTLN scheduler context is nil")
	}

	s.stateMu.Lock()
	defer s.stateMu.Unlock()

	if s.closed {
		return errStreamClosed
	}
	if s.inFlight.Load() {
		return errStreamProcessing
	}
	if s.scheduler.closed.Load() {
		return errSchedulerClosed
	}

	<-s.idle
	s.inFlight.Store(true)

	return nil
}

// finishProcess marks the stream idle and wakes lifecycle waiters
func (s *ScheduledStream) finishProcess() {
	s.stateMu.Lock()
	s.inFlight.Store(false)
	s.stateMu.Unlock()
	s.idle <- struct{}{}
}

// prepareRequest populates reusable request storage with an effective deadline
func (s *ScheduledStream) prepareRequest(
	ctx context.Context, deadline time.Time, operation scheduledOperation, dst, samples []float32,
) *scheduledRequest {
	if contextDeadline, ok := ctx.Deadline(); ok && (deadline.IsZero() || contextDeadline.Before(deadline)) {
		deadline = contextDeadline
	}

	request := &s.request
	request.context = ctx
	request.processor = s.processor
	request.deadline = deadline
	request.enqueued = time.Now()
	request.operation = operation
	request.dst = dst
	request.samples = samples
	request.result = s.result

	return request
}

// Drain waits until the stream has no queued or active operation
func (s *ScheduledStream) Drain(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("DTLN scheduler context is nil")
	}

	return s.waitForIdle(ctx)
}

// Close rejects new stream operations, drains active work, and closes the processor
func (s *ScheduledStream) Close(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("DTLN scheduler context is nil")
	}

	s.stateMu.Lock()
	s.closed = true
	s.stateMu.Unlock()

	if err := s.waitForIdle(ctx); err != nil {
		return err
	}

	s.closeOnce.Do(func() {
		s.closeErr = s.processor.Close()
		close(s.closeDone)
	})

	select {
	case <-s.closeDone:
		return s.closeErr
	case <-ctx.Done():
		return ctx.Err()
	}
}

// waitForIdle waits for the reusable idle token when work is active
func (s *ScheduledStream) waitForIdle(ctx context.Context) error {
	s.stateMu.Lock()
	inFlight := s.inFlight.Load()
	s.stateMu.Unlock()
	if !inFlight {
		return nil
	}

	select {
	case <-s.idle:
		s.idle <- struct{}{}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// submit transfers one prepared request to the scheduler dispatcher
func (s *ScheduledStream) submit(ctx context.Context, request *scheduledRequest) error {
	select {
	case s.scheduler.requests <- request:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-s.scheduler.done:
		return errSchedulerClosed
	}
}

// awaitResult waits for worker completion or scheduler shutdown
func (s *ScheduledStream) awaitResult() ([]float32, ScheduleMetrics, error) {
	select {
	case result := <-s.result:
		return result.output, result.metrics, result.err
	case <-s.scheduler.done:
		select {
		case result := <-s.result:
			return result.output, result.metrics, result.err
		default:
			return nil, ScheduleMetrics{}, errSchedulerClosed
		}
	}
}

// Close rejects queued work and waits for active inference calls to finish
func (s *Scheduler) Close() {
	if s == nil {
		return
	}

	if !s.closed.CompareAndSwap(false, true) {
		<-s.done
		return
	}

	closed := make(chan struct{})
	select {
	case s.closeRequests <- closed:
		<-closed
	case <-s.done:
	}
}

// dispatch owns queue ordering and active processor tracking
func (s *Scheduler) dispatch(workerCount int) {
	defer close(s.done)

	queue := scheduledQueue{}
	heap.Init(&queue)
	active := make(map[*Processor]bool)
	activeCount := 0
	var sequence uint64
	var closeComplete chan struct{}

	for {
		activeCount += s.startReadyRequests(&queue, active, workerCount-activeCount, closeComplete != nil)

		if closeComplete != nil && activeCount == 0 {
			close(s.work)
			close(closeComplete)
			return
		}

		select {
		case request := <-s.requests:
			sequence = s.enqueueRequest(&queue, request, sequence)
		case completion := <-s.completed:
			delete(active, completion.request.processor)
			activeCount--
			completion.request.result <- completion.result
		case closeComplete = <-s.closeRequests:
			for queue.Len() > 0 {
				request := heap.Pop(&queue).(*scheduledRequest)
				request.result <- scheduledResult{err: errSchedulerClosed}
			}
		}
	}
}

// startReadyRequests assigns eligible requests to available workers
func (s *Scheduler) startReadyRequests(
	queue *scheduledQueue,
	active map[*Processor]bool,
	availableWorkers int,
	closing bool,
) int {
	if closing {
		return 0
	}

	started := 0
	for started < availableWorkers {
		request := popReadyRequest(queue, active)
		if request == nil {
			break
		}
		if err := request.context.Err(); err != nil {
			request.result <- scheduledResult{err: err}
			continue
		}

		active[request.processor] = true
		started++
		s.work <- request
	}

	return started
}

// enqueueRequest records submission order and inserts one request by priority
func (s *Scheduler) enqueueRequest(
	queue *scheduledQueue,
	request *scheduledRequest,
	sequence uint64,
) uint64 {
	if s.closed.Load() {
		request.result <- scheduledResult{err: errSchedulerClosed}
		return sequence
	}

	request.sequence = sequence
	heap.Push(queue, request)

	return sequence + 1
}

// worker executes requests until scheduler shutdown
func (s *Scheduler) worker() {
	for request := range s.work {
		s.execute(request)
	}
}

// execute measures and reports one processor operation
func (s *Scheduler) execute(request *scheduledRequest) {
	started := time.Now()
	output, err := executeScheduledOperation(request)
	finished := time.Now()

	s.completed <- scheduledCompletion{
		request: request,
		result: scheduledResult{
			output: output,
			metrics: ScheduleMetrics{
				QueueDuration:      started.Sub(request.enqueued),
				ProcessingDuration: finished.Sub(started),
				DeadlineMissed:     !request.deadline.IsZero() && finished.After(request.deadline),
			},
			err: err,
		},
	}
}

// executeScheduledOperation dispatches process and flush requests to a processor
func executeScheduledOperation(request *scheduledRequest) ([]float32, error) {
	if request.operation == flushOperation {
		return request.processor.FlushInto(request.dst)
	}

	return request.processor.ProcessInto(request.dst, request.samples)
}

// popReadyRequest removes the highest priority request for an idle processor
func popReadyRequest(queue *scheduledQueue, active map[*Processor]bool) *scheduledRequest {
	readyIndex := -1

	for index, request := range *queue {
		if active[request.processor] {
			continue
		}

		if readyIndex < 0 || queue.Less(index, readyIndex) {
			readyIndex = index
		}
	}

	if readyIndex < 0 {
		return nil
	}

	return heap.Remove(queue, readyIndex).(*scheduledRequest)
}
