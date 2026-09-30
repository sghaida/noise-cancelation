package dtln

import (
	"container/heap"
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

type blockingModelRunner struct {
	output  []float32
	started chan struct{}
	release chan struct{}
	once    sync.Once
}

func (m *blockingModelRunner) run([]float32) ([]float32, error) {
	m.once.Do(func() { close(m.started) })
	<-m.release

	return m.output, nil
}

func (m *blockingModelRunner) reset() {}

func (m *blockingModelRunner) close() error { return nil }

func TestNewSchedulerValidation(t *testing.T) {
	if _, err := NewScheduler(0); err == nil {
		t.Fatal("NewScheduler() accepted zero workers")
	}
}

func TestScheduledQueueOrdersDeadlinesThenSequence(t *testing.T) {
	now := time.Now()
	queue := scheduledQueue{
		&scheduledRequest{deadline: time.Time{}, sequence: 0},
		&scheduledRequest{deadline: now.Add(time.Second), sequence: 2},
		&scheduledRequest{deadline: now.Add(time.Second), sequence: 1},
		&scheduledRequest{deadline: now.Add(2 * time.Second), sequence: 0},
	}
	heap.Init(&queue)

	wantSequences := []uint64{1, 2, 0, 0}
	for index, want := range wantSequences {
		request := heap.Pop(&queue).(*scheduledRequest)
		if request.sequence != want {
			t.Fatalf("queue item %d sequence = %d, want %d", index, request.sequence, want)
		}
	}
}

func TestSchedulerProcess(t *testing.T) {
	scheduler, err := NewScheduler(1)
	if err != nil {
		t.Fatalf("NewScheduler() error: %v", err)
	}
	defer scheduler.Close()

	mask := make([]float32, SpectrumBins)
	for i := range mask {
		mask[i] = 1
	}

	processor := newTestProcessor(
		&fakeModelRunner{output: mask},
		&fakeModelRunner{output: make([]float32, BlockLength)},
	)
	samples := make([]float32, BlockShift)
	dst := make([]float32, 0, BlockShift)
	stream, err := scheduler.NewStream(processor)
	if err != nil {
		t.Fatalf("NewStream() error: %v", err)
	}

	output, metrics, err := stream.Process(
		context.Background(),
		time.Now().Add(time.Second),
		dst,
		samples,
	)
	if err != nil {
		t.Fatalf("Process() error: %v", err)
	}

	if len(output) != BlockShift {
		t.Fatalf("Process() output length = %d, want %d", len(output), BlockShift)
	}
	if metrics.QueueDuration < 0 || metrics.ProcessingDuration <= 0 {
		t.Fatalf("Process() metrics = %#v", metrics)
	}
	if metrics.DeadlineMissed {
		t.Fatal("Process() unexpectedly missed deadline")
	}
}

func TestScheduledStreamProcessAllocs(t *testing.T) {
	scheduler, err := NewScheduler(1)
	if err != nil {
		t.Fatalf("NewScheduler() error: %v", err)
	}
	defer scheduler.Close()

	processor := newTestProcessor(
		&fakeModelRunner{output: make([]float32, SpectrumBins)},
		&fakeModelRunner{output: make([]float32, BlockLength)},
	)
	stream, err := scheduler.NewStream(processor)
	if err != nil {
		t.Fatalf("NewStream() error: %v", err)
	}
	defer func() { _ = stream.Close(context.Background()) }()

	samples := make([]float32, BlockShift)
	output := make([]float32, 0, BlockShift)
	if _, _, err := stream.Process(context.Background(), time.Time{}, output[:0], samples); err != nil {
		t.Fatalf("warm-up Process() error: %v", err)
	}

	allocations := testing.AllocsPerRun(100, func() {
		if _, _, err := stream.Process(context.Background(), time.Time{}, output[:0], samples); err != nil {
			t.Fatalf("Process() error: %v", err)
		}
	})

	if allocations != 0 {
		t.Fatalf("scheduled Process() allocations = %v, want 0", allocations)
	}
}

func TestSchedulerReportsMissedDeadline(t *testing.T) {
	scheduler, err := NewScheduler(1)
	if err != nil {
		t.Fatalf("NewScheduler() error: %v", err)
	}
	defer scheduler.Close()

	processor := newTestProcessor(
		&fakeModelRunner{output: make([]float32, SpectrumBins)},
		&fakeModelRunner{output: make([]float32, BlockLength)},
	)
	stream, err := scheduler.NewStream(processor)
	if err != nil {
		t.Fatalf("NewStream() error: %v", err)
	}

	_, metrics, err := stream.Process(
		context.Background(),
		time.Now().Add(-time.Second),
		make([]float32, 0, BlockShift),
		make([]float32, BlockShift),
	)
	if err != nil {
		t.Fatalf("Process() error: %v", err)
	}
	if !metrics.DeadlineMissed {
		t.Fatal("Process() did not report missed deadline")
	}
}

func newScheduledFlushTestStream(t *testing.T) (*ScheduledStream, *Processor) {
	t.Helper()

	scheduler, err := NewScheduler(1)
	if err != nil {
		t.Fatalf("NewScheduler() error: %v", err)
	}
	t.Cleanup(scheduler.Close)

	processor := newTestProcessor(
		&fakeModelRunner{output: make([]float32, SpectrumBins)},
		&fakeModelRunner{output: make([]float32, BlockLength)},
	)
	stream, err := scheduler.NewStream(processor)
	if err != nil {
		t.Fatalf("NewStream() error: %v", err)
	}
	t.Cleanup(func() { _ = stream.Close(context.Background()) })

	return stream, processor
}

func prepareScheduledFlushTest(t *testing.T, stream *ScheduledStream) {
	t.Helper()

	processOutput := make([]float32, 0, BlockShift)
	if _, _, err := stream.Process(
		context.Background(),
		time.Time{},
		processOutput,
		make([]float32, BlockShift+37),
	); err != nil {
		t.Fatalf("Process() error: %v", err)
	}
}

func TestScheduledStreamFlushCapacityError(t *testing.T) {
	stream, processor := newScheduledFlushTestStream(t)
	prepareScheduledFlushTest(t, stream)

	if _, _, err := stream.Flush(context.Background(), time.Time{}, nil); err == nil {
		t.Fatal("Flush() accepted insufficient output capacity")
	}
	if processor.pendingCount != 37 {
		t.Fatalf("pending count after flush capacity error = %d, want 37", processor.pendingCount)
	}
}

func TestScheduledStreamFlush(t *testing.T) {
	stream, _ := newScheduledFlushTestStream(t)
	prepareScheduledFlushTest(t, stream)

	flushOutput := make([]float32, 0, 37)
	output, metrics, err := stream.Flush(
		context.Background(),
		time.Now().Add(time.Second),
		flushOutput,
	)
	if err != nil {
		t.Fatalf("Flush() error: %v", err)
	}
	if len(output) != 37 {
		t.Fatalf("Flush() output length = %d, want 37", len(output))
	}
	if metrics.QueueDuration < 0 || metrics.ProcessingDuration <= 0 || metrics.DeadlineMissed {
		t.Fatalf("Flush() metrics = %#v", metrics)
	}

	second, _, err := stream.Flush(context.Background(), time.Time{}, flushOutput[:0])
	if err != nil {
		t.Fatalf("second Flush() error: %v", err)
	}
	if len(second) != 0 {
		t.Fatalf("second Flush() length = %d, want 0", len(second))
	}
}

func TestScheduledStreamClose(t *testing.T) {
	scheduler, err := NewScheduler(1)
	if err != nil {
		t.Fatalf("NewScheduler() error: %v", err)
	}
	defer scheduler.Close()

	model1 := &fakeModelRunner{output: make([]float32, SpectrumBins)}
	model2 := &fakeModelRunner{output: make([]float32, BlockLength)}
	stream, err := scheduler.NewStream(newTestProcessor(model1, model2))
	if err != nil {
		t.Fatalf("NewStream() error: %v", err)
	}

	if err := stream.Close(context.Background()); err != nil {
		t.Fatalf("Close() error: %v", err)
	}
	if err := stream.Close(context.Background()); err != nil {
		t.Fatalf("second Close() error: %v", err)
	}
	if model1.closeCount != 1 || model2.closeCount != 1 {
		t.Fatalf("model close counts = %d and %d, want 1 and 1", model1.closeCount, model2.closeCount)
	}
	if _, _, err := stream.Process(context.Background(), time.Time{}, nil, nil); !errors.Is(err, errStreamClosed) {
		t.Fatalf("closed Process() error = %v, want stream closed", err)
	}
	if _, _, err := stream.Flush(context.Background(), time.Time{}, nil); !errors.Is(err, errStreamClosed) {
		t.Fatalf("closed Flush() error = %v, want stream closed", err)
	}
}

func TestScheduledStreamDrainAndCloseWaitForActiveWork(t *testing.T) {
	scheduler, err := NewScheduler(1)
	if err != nil {
		t.Fatalf("NewScheduler() error: %v", err)
	}
	defer scheduler.Close()

	model := &blockingModelRunner{
		output:  make([]float32, SpectrumBins),
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	model2 := &fakeModelRunner{output: make([]float32, BlockLength)}
	processor := newTestProcessor(model, model2)
	stream, err := scheduler.NewStream(processor)
	if err != nil {
		t.Fatalf("NewStream() error: %v", err)
	}

	processDone := make(chan error, 1)
	go func() {
		_, _, processErr := stream.Process(
			context.Background(),
			time.Time{},
			make([]float32, 0, BlockShift),
			make([]float32, BlockShift),
		)
		processDone <- processErr
	}()
	<-model.started

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := stream.Drain(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled Drain() error = %v, want context canceled", err)
	}

	closeDone := make(chan error, 1)
	go func() { closeDone <- stream.Close(context.Background()) }()
	select {
	case err := <-closeDone:
		t.Fatalf("Close() returned during active inference: %v", err)
	default:
	}

	close(model.release)
	if err := <-processDone; err != nil {
		t.Fatalf("Process() error: %v", err)
	}
	if err := <-closeDone; err != nil {
		t.Fatalf("Close() error: %v", err)
	}
	if err := stream.Drain(context.Background()); err != nil {
		t.Fatalf("Drain() after close error: %v", err)
	}
	if model2.closeCount != 1 {
		t.Fatalf("model 2 close count = %d, want 1", model2.closeCount)
	}
}

func TestScheduledStreamCloseCanResumeAfterCancellation(t *testing.T) {
	scheduler, err := NewScheduler(1)
	if err != nil {
		t.Fatalf("NewScheduler() error: %v", err)
	}
	defer scheduler.Close()

	model := &blockingModelRunner{
		output:  make([]float32, SpectrumBins),
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	processor := newTestProcessor(model, &fakeModelRunner{output: make([]float32, BlockLength)})
	stream, err := scheduler.NewStream(processor)
	if err != nil {
		t.Fatalf("NewStream() error: %v", err)
	}

	processDone := make(chan error, 1)
	go func() {
		_, _, processErr := stream.Process(
			context.Background(),
			time.Time{},
			make([]float32, 0, BlockShift),
			make([]float32, BlockShift),
		)
		processDone <- processErr
	}()
	<-model.started

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := stream.Close(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled Close() error = %v, want context canceled", err)
	}
	if _, _, err := stream.Process(context.Background(), time.Time{}, nil, nil); !errors.Is(err, errStreamClosed) {
		t.Fatalf("Process() after cancelled close error = %v, want stream closed", err)
	}

	close(model.release)
	if err := <-processDone; err != nil {
		t.Fatalf("Process() error: %v", err)
	}
	if err := stream.Close(context.Background()); err != nil {
		t.Fatalf("resumed Close() error: %v", err)
	}
}

func TestScheduledStreamRejectsConcurrentChunk(t *testing.T) {
	scheduler, err := NewScheduler(1)
	if err != nil {
		t.Fatalf("NewScheduler() error: %v", err)
	}
	defer scheduler.Close()

	model := &blockingModelRunner{
		output:  make([]float32, SpectrumBins),
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	processor := newTestProcessor(model, &fakeModelRunner{output: make([]float32, BlockLength)})
	stream, err := scheduler.NewStream(processor)
	if err != nil {
		t.Fatalf("NewStream() error: %v", err)
	}

	finished := make(chan error, 1)
	go func() {
		_, _, processErr := stream.Process(
			context.Background(),
			time.Time{},
			make([]float32, 0, BlockShift),
			make([]float32, BlockShift),
		)
		finished <- processErr
	}()

	<-model.started
	if _, _, err := stream.Process(context.Background(), time.Time{}, nil, nil); !errors.Is(err, errStreamProcessing) {
		t.Fatalf("concurrent Process() error = %v, want stream processing", err)
	}

	close(model.release)
	if err := <-finished; err != nil {
		t.Fatalf("first Process() error: %v", err)
	}
}

func TestSchedulerCloseDrainsActiveAndRejectsQueued(t *testing.T) {
	scheduler, err := NewScheduler(1)
	if err != nil {
		t.Fatalf("NewScheduler() error: %v", err)
	}

	model := &blockingModelRunner{
		output:  make([]float32, SpectrumBins),
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	activeProcessor := newTestProcessor(model, &fakeModelRunner{output: make([]float32, BlockLength)})
	activeStream, err := scheduler.NewStream(activeProcessor)
	if err != nil {
		t.Fatalf("active NewStream() error: %v", err)
	}
	queuedStream, err := scheduler.NewStream(newTestProcessor(&fakeModelRunner{}, &fakeModelRunner{}))
	if err != nil {
		t.Fatalf("queued NewStream() error: %v", err)
	}

	activeDone := make(chan error, 1)
	go func() {
		_, _, processErr := activeStream.Process(
			context.Background(),
			time.Time{},
			make([]float32, 0, BlockShift),
			make([]float32, BlockShift),
		)
		activeDone <- processErr
	}()
	<-model.started

	queuedDone := make(chan error, 1)
	go func() {
		_, _, processErr := queuedStream.Process(context.Background(), time.Time{}, nil, nil)
		queuedDone <- processErr
	}()
	for !queuedStream.inFlight.Load() {
	}

	closeDone := make(chan struct{})
	go func() {
		scheduler.Close()
		close(closeDone)
	}()

	if err := <-queuedDone; !errors.Is(err, errSchedulerClosed) {
		t.Fatalf("queued Process() error = %v, want scheduler closed", err)
	}
	select {
	case <-closeDone:
		t.Fatal("Close() returned while inference was active")
	default:
	}

	close(model.release)
	if err := <-activeDone; err != nil {
		t.Fatalf("active Process() error: %v", err)
	}
	<-closeDone
}

func TestSchedulerRejectsCancelledQueuedRequest(t *testing.T) {
	scheduler, err := NewScheduler(1)
	if err != nil {
		t.Fatalf("NewScheduler() error: %v", err)
	}
	defer scheduler.Close()

	model := &blockingModelRunner{
		output:  make([]float32, SpectrumBins),
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	activeProcessor := newTestProcessor(model, &fakeModelRunner{output: make([]float32, BlockLength)})
	activeStream, err := scheduler.NewStream(activeProcessor)
	if err != nil {
		t.Fatalf("NewStream() error: %v", err)
	}

	activeDone := make(chan error, 1)
	go func() {
		_, _, processErr := activeStream.Process(
			context.Background(),
			time.Time{},
			make([]float32, 0, BlockShift),
			make([]float32, BlockShift),
		)
		activeDone <- processErr
	}()
	<-model.started

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	result := make(chan scheduledResult, 1)
	scheduler.requests <- &scheduledRequest{
		context:   cancelled,
		processor: newTestProcessor(&fakeModelRunner{}, &fakeModelRunner{}),
		result:    result,
	}
	close(model.release)

	if scheduled := <-result; !errors.Is(scheduled.err, context.Canceled) {
		t.Fatalf("queued request error = %v, want context canceled", scheduled.err)
	}
	if err := <-activeDone; err != nil {
		t.Fatalf("active Process() error: %v", err)
	}
}

func TestSchedulerRejectsInvalidAndClosedRequests(t *testing.T) {
	scheduler, err := NewScheduler(1)
	if err != nil {
		t.Fatalf("NewScheduler() error: %v", err)
	}

	if _, err := scheduler.NewStream(nil); err == nil {
		t.Fatal("NewStream() accepted nil processor")
	}

	processor := newTestProcessor(&fakeModelRunner{}, &fakeModelRunner{})
	stream, err := scheduler.NewStream(processor)
	if err != nil {
		t.Fatalf("NewStream() error: %v", err)
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := stream.Process(cancelled, time.Time{}, nil, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("Process() error = %v, want context canceled", err)
	}

	scheduler.Close()
	if _, _, err := stream.Process(context.Background(), time.Time{}, nil, nil); !errors.Is(err, errSchedulerClosed) {
		t.Fatalf("closed Process() error = %v, want scheduler closed", err)
	}
	if _, err := scheduler.NewStream(processor); !errors.Is(err, errSchedulerClosed) {
		t.Fatalf("closed NewStream() error = %v, want scheduler closed", err)
	}
}

func TestSchedulerCloseIsIdempotent(t *testing.T) {
	var nilScheduler *Scheduler
	nilScheduler.Close()

	scheduler, err := NewScheduler(1)
	if err != nil {
		t.Fatalf("NewScheduler() error: %v", err)
	}

	scheduler.Close()
	scheduler.Close()
}
