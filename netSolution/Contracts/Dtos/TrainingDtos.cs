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
    string? MetricsJson,
    bool CancellationRequested = false,
    string? ClientId = null,
    DateTime? AssignedAt = null,
    DateTime? HeartbeatAt = null,
    string? DatasetZipPath = null,
    int? Epochs = null,
    int? ImgSize = null,
    int? Batch = null,
    string? Device = null,
    bool? ExportInt8 = null,
    bool? ExportNms = null,
    double? QuantizationFraction = null,
    string? MobileModelFileName = null,
    string? MobileModelContentType = null
);

public record TrainingClientJobResponse(
    string JobId,
    int ImagesCount,
    string? BaseModel,
    int Epochs,
    int ImgSize,
    int Batch,
    string? Device,
    bool ExportInt8,
    bool ExportNms,
    string? MobileFormat,
    double QuantizationFraction,
    bool CancellationRequested
);

public record TrainingClientStatusRequest(
    string Status,
    string? Message,
    string? MetricsJson = null
);

public record TrainingClientStatusResponse(
    string JobId,
    string Status,
    bool CancellationRequested,
    string? Message
);

public record ModelVersionAdminResponse(
    int Id,
    string? ExternalJobId,
    DateTime TrainedAt,
    string? MetricsJson,
    string? BaseModel,
    string? BestWeightsPath,
    string? MobileModelPath,
    string? MobileModelFileName,
    string? MobileModelContentType,
    string? MobileFormat,
    bool IsPublished,
    bool IsPinned,
    bool IsDeleted,
    DateTime? DeletedAt = null
);

public record LatestMobileModelResponse(
    int ModelVersionId,
    DateTime TrainedAt,
    string? MobileFormat,
    string? MetricsJson,
    string? FileName = null,
    bool IsPinned = false
);
