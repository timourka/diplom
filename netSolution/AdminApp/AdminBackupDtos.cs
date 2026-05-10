namespace AdminApp;

public sealed record BackupImportResultDto(
    int Users,
    int Products,
    int StoredProducts,
    int VideoSamples,
    int ErrorReports,
    int ModelVersions,
    int TrainingJobs,
    bool ReplacedExisting);
