namespace Contracts.Dtos;

public record StartTrainingRequest(
    string? BaseModel,
    int? Epochs,
    int? ImgSize,
    int? Batch,
    string? Device,
    bool? ExportInt8,
    bool? ExportNms,
    string? MobileFormat,
    double? QuantizationFraction
);

public record TrainingJobStartResponse(
    string JobId,
    string Status,
    int ImagesCount,
    string Message
);

public record TrainingJobStatusResponse(
    string JobId,
    string Status,
    string? Message,
    DateTime CreatedAt,
    DateTime? StartedAt,
    DateTime? FinishedAt,
    int ImagesCount,
    string? BaseModel,
    string? BestWeightsPath,
    string? MobileModelPath,
    string? MobileFormat,
    string? MetricsJson
);

public record LatestMobileModelResponse(
    int ModelVersionId,
    DateTime TrainedAt,
    string? MobileFormat,
    string? MetricsJson
);
