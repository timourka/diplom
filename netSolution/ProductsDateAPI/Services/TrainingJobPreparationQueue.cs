using System.Threading.Channels;

namespace ProductsDateAPI.Services;

public interface ITrainingJobPreparationQueue
{
    void Enqueue(string jobId);
    ValueTask<string?> DequeueAsync(TimeSpan timeout, CancellationToken ct);
}

public sealed class TrainingJobPreparationQueue : ITrainingJobPreparationQueue
{
    private readonly Channel<string> _channel = Channel.CreateUnbounded<string>(new UnboundedChannelOptions
    {
        SingleReader = true,
        SingleWriter = false,
    });

    public void Enqueue(string jobId)
    {
        if (!string.IsNullOrWhiteSpace(jobId))
            _channel.Writer.TryWrite(jobId);
    }

    public async ValueTask<string?> DequeueAsync(TimeSpan timeout, CancellationToken ct)
    {
        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeoutCts.CancelAfter(timeout);

        try
        {
            if (await _channel.Reader.WaitToReadAsync(timeoutCts.Token))
            {
                if (_channel.Reader.TryRead(out var jobId))
                    return jobId;
            }
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            return null;
        }

        return null;
    }
}
