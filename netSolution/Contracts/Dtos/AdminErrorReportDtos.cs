namespace Contracts.Dtos;

public record AdminErrorReportListItemDto(
    int Id,
    int UserId,
    DateTime CreatedAt,
    int FramesCount,
    bool Approved
);

public record AdminErrorReportDetailsDto(
    int Id,
    int UserId,
    DateTime CreatedAt,
    int FramesCount,
    bool Approved,
    string? Comment,
    string DatasetFolder
);

public record ApproveErrorReportRequest(bool Approved);

public record YoloBboxDto(
    int ClassId,
    double Xc,
    double Yc,
    double W,
    double H
);