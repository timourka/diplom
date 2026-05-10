using Contracts.Dtos;
using System.Globalization;
using System.Text.Json;

namespace AdminApp;

public sealed class FormModelVersionDetails : Form
{
    private readonly AdminApiClient _api;
    private readonly int _modelVersionId;
    private ModelVersionAdminResponse? _model;

    private readonly Label _lblIdValue = CreateValueLabel();
    private readonly Label _lblStateValue = CreateValueLabel();
    private readonly Label _lblTrainedAtValue = CreateValueLabel();
    private readonly Label _lblDeletedAtValue = CreateValueLabel();
    private readonly Label _lblBaseModelValue = CreateValueLabel();
    private readonly Label _lblMobileFormatValue = CreateValueLabel();
    private readonly Label _lblJobIdValue = CreateValueLabel();
    private readonly Label _lblContentTypeValue = CreateValueLabel();

    private readonly DataGridView _gridMetrics = new()
    {
        Dock = DockStyle.Fill,
        ReadOnly = true,
        AllowUserToAddRows = false,
        AllowUserToDeleteRows = false,
        AutoGenerateColumns = false,
        SelectionMode = DataGridViewSelectionMode.FullRowSelect,
        MultiSelect = false,
        RowHeadersVisible = false,
        AutoSizeRowsMode = DataGridViewAutoSizeRowsMode.AllCells,
        DefaultCellStyle = new DataGridViewCellStyle { WrapMode = DataGridViewTriState.True },
    };

    private readonly TextBox _txtFiles = new()
    {
        Multiline = true,
        ReadOnly = true,
        ScrollBars = ScrollBars.Vertical,
        Dock = DockStyle.Fill,
    };

    private readonly TextBox _txtRelatedJob = new()
    {
        Multiline = true,
        ReadOnly = true,
        ScrollBars = ScrollBars.Vertical,
        Dock = DockStyle.Fill,
    };

    private readonly Button _btnRefresh = new() { Text = "Обновить", Width = 120, Height = 36 };
    private readonly Button _btnOpenJob = new() { Text = "Открыть задачу обучения", Width = 210, Height = 36 };
    private readonly Button _btnClose = new() { Text = "Закрыть", Width = 120, Height = 36 };

    public FormModelVersionDetails(AdminApiClient api, ModelVersionAdminResponse model)
        : this(api, model.Id)
    {
        _model = model;
    }

    public FormModelVersionDetails(AdminApiClient api, int modelVersionId)
    {
        _api = api;
        _modelVersionId = modelVersionId;

        Text = $"Версия модели #{modelVersionId}";
        Width = 1120;
        Height = 780;
        StartPosition = FormStartPosition.CenterParent;

        BuildUi();

        _btnRefresh.Click += async (_, _) => await LoadModelAsync();
        _btnOpenJob.Click += (_, _) => OpenRelatedJob();
        _btnClose.Click += (_, _) => Close();
        Shown += async (_, _) => await LoadModelAsync();
    }

    private void BuildUi()
    {
        ConfigureMetricsGrid();

        var header = new TableLayoutPanel
        {
            Dock = DockStyle.Top,
            AutoSize = true,
            Padding = new Padding(12),
            ColumnCount = 4,
        };

        header.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 170));
        header.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
        header.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 170));
        header.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));

        AddInfoRow(header, 0, "ID версии", _lblIdValue, "Состояние", _lblStateValue);
        AddInfoRow(header, 1, "Обучена", _lblTrainedAtValue, "Удалена", _lblDeletedAtValue);
        AddInfoRow(header, 2, "Базовая модель", _lblBaseModelValue, "Mobile формат", _lblMobileFormatValue);
        AddInfoRow(header, 3, "Job ID", _lblJobIdValue, "Content-Type", _lblContentTypeValue);

        var actions = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            AutoSize = true,
            Padding = new Padding(12, 0, 12, 12),
            FlowDirection = FlowDirection.LeftToRight,
        };
        actions.Controls.AddRange([_btnRefresh, _btnOpenJob, _btnClose]);

        var content = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            Padding = new Padding(12, 0, 12, 12),
            ColumnCount = 1,
            RowCount = 6,
        };
        content.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        content.RowStyles.Add(new RowStyle(SizeType.Percent, 52));
        content.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        content.RowStyles.Add(new RowStyle(SizeType.Percent, 24));
        content.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        content.RowStyles.Add(new RowStyle(SizeType.Percent, 24));

        content.Controls.Add(CreateSectionLabel("Метрики качества"), 0, 0);
        content.Controls.Add(_gridMetrics, 0, 1);
        content.Controls.Add(CreateSectionLabel("Файлы модели на backend"), 0, 2);
        content.Controls.Add(_txtFiles, 0, 3);
        content.Controls.Add(CreateSectionLabel("Связанная задача обучения"), 0, 4);
        content.Controls.Add(_txtRelatedJob, 0, 5);

        Controls.Add(content);
        Controls.Add(actions);
        Controls.Add(header);
    }

    private void ConfigureMetricsGrid()
    {
        _gridMetrics.Columns.Add(new DataGridViewTextBoxColumn
        {
            HeaderText = "Метрика",
            DataPropertyName = nameof(MetricRow.Name),
            Width = 190
        });
        _gridMetrics.Columns.Add(new DataGridViewTextBoxColumn
        {
            HeaderText = "Значение",
            DataPropertyName = nameof(MetricRow.Value),
            Width = 140
        });
        _gridMetrics.Columns.Add(new DataGridViewTextBoxColumn
        {
            HeaderText = "Пояснение",
            DataPropertyName = nameof(MetricRow.Description),
            AutoSizeMode = DataGridViewAutoSizeColumnMode.Fill
        });
    }

    private async Task LoadModelAsync()
    {
        try
        {
            ToggleBusy(true);

            var versions = await _api.GetModelVersionsAsync();
            var latest = versions.FirstOrDefault(x => x.Id == _modelVersionId);
            if (latest is not null)
                _model = latest;

            if (_model is null)
            {
                MessageBox.Show(this, "Версия модели не найдена на сервере.", "Версия модели", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                Close();
                return;
            }

            Render(_model);
            await LoadRelatedJobAsync(_model.ExternalJobId);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка загрузки версии модели", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            ToggleBusy(false);
        }
    }

    private void Render(ModelVersionAdminResponse model)
    {
        Text = $"Версия модели #{model.Id}";
        _lblIdValue.Text = model.Id.ToString(CultureInfo.InvariantCulture);
        _lblStateValue.Text = BuildState(model);
        _lblTrainedAtValue.Text = ToLocalString(model.TrainedAt);
        _lblDeletedAtValue.Text = ToLocalString(model.DeletedAt);
        _lblBaseModelValue.Text = model.BaseModel ?? "-";
        _lblMobileFormatValue.Text = model.MobileFormat ?? "-";
        _lblJobIdValue.Text = model.ExternalJobId ?? "-";
        _lblContentTypeValue.Text = model.MobileModelContentType ?? "-";

        _txtFiles.Text = string.Join(Environment.NewLine, new[]
        {
            $"Mobile file name: {model.MobileModelFileName ?? "-"}",
            $"Mobile model path: {model.MobileModelPath ?? "-"}",
            $"Best weights path: {model.BestWeightsPath ?? "-"}",
            $"Content-Type: {model.MobileModelContentType ?? "-"}"
        });

        _gridMetrics.DataSource = BuildMetricRows(model.MetricsJson);
        _btnOpenJob.Enabled = !string.IsNullOrWhiteSpace(model.ExternalJobId);
    }

    private async Task LoadRelatedJobAsync(string? jobId)
    {
        if (string.IsNullOrWhiteSpace(jobId))
        {
            _txtRelatedJob.Text = "У этой версии нет связанной задачи обучения.";
            return;
        }

        try
        {
            var job = await _api.GetTrainingJobAsync(jobId);
            if (job is null)
            {
                _txtRelatedJob.Text = $"Связанная задача {jobId} не найдена.";
                return;
            }

            _txtRelatedJob.Text = BuildJobSummary(job);
        }
        catch (Exception ex)
        {
            _txtRelatedJob.Text = $"Не удалось загрузить связанную задачу {jobId}." + Environment.NewLine + ex.Message;
        }
    }

    private void OpenRelatedJob()
    {
        var jobId = _model?.ExternalJobId;
        if (string.IsNullOrWhiteSpace(jobId))
            return;

        using var form = new FormTrainingJobDetails(_api, jobId);
        form.ShowDialog(this);
    }

    private void ToggleBusy(bool busy)
    {
        UseWaitCursor = busy;
        _btnRefresh.Enabled = !busy;
        _btnClose.Enabled = !busy;
        _btnOpenJob.Enabled = !busy && !string.IsNullOrWhiteSpace(_model?.ExternalJobId);
    }

    private static string BuildState(ModelVersionAdminResponse model)
    {
        var parts = new List<string>();
        if (model.IsPublished) parts.Add("текущая для пользователей");
        if (model.IsPinned) parts.Add("зафиксирована");
        if (model.IsDeleted) parts.Add("удалена");
        return parts.Count == 0 ? "обычная версия" : string.Join(", ", parts);
    }

    private static string BuildJobSummary(TrainingJobStatusResponse job)
        => string.Join(Environment.NewLine, new[]
        {
            $"Job ID: {job.JobId}",
            $"Статус: {TranslateStatus(job.Status)}",
            $"Клиент обучения: {job.ClientId ?? "-"}",
            $"Создано: {ToLocalString(job.CreatedAt)}",
            $"Назначено клиенту: {ToLocalString(job.AssignedAt)}",
            $"Запущено: {ToLocalString(job.StartedAt)}",
            $"Последний heartbeat: {ToLocalString(job.HeartbeatAt)}",
            $"Завершено: {ToLocalString(job.FinishedAt)}",
            $"Датасет кадров: {job.ImagesCount}",
            $"Параметры: epochs={job.Epochs?.ToString(CultureInfo.InvariantCulture) ?? "-"}, imgsz={job.ImgSize?.ToString(CultureInfo.InvariantCulture) ?? "-"}, batch={job.Batch?.ToString(CultureInfo.InvariantCulture) ?? "-"}, device={job.Device ?? "-"}",
            $"Mobile export: format={job.MobileFormat ?? "-"}, int8={BoolText(job.ExportInt8)}, nms={BoolText(job.ExportNms)}, fraction={job.QuantizationFraction?.ToString("0.###", CultureInfo.InvariantCulture) ?? "-"}",
            $"Сообщение: {job.Message ?? "-"}"
        });

    private static List<MetricRow> BuildMetricRows(string? metricsJson)
    {
        if (string.IsNullOrWhiteSpace(metricsJson))
        {
            return
            [
                new MetricRow
                {
                    Name = "Метрики не сохранены",
                    Value = "-",
                    Description = "Для этой версии модели backend не получил JSON метрик от клиента обучения."
                }
            ];
        }

        try
        {
            using var document = JsonDocument.Parse(metricsJson);
            var root = document.RootElement;
            var rows = new List<MetricRow>();

            AddMetric(rows, root, "mAP50_95", "mAP@0.50:0.95", "Главная итоговая метрика детектора. Среднее качество обнаружения на разных уровнях IoU от 0.50 до 0.95. Чем выше, тем лучше.");
            AddMetric(rows, root, "mAP50", "mAP@0.50", "Качество обнаружения при мягком критерии совпадения рамок. Чем выше, тем лучше.");
            AddMetric(rows, root, "mAP75", "mAP@0.75", "Качество обнаружения при более строгом сравнении рамок. Чем выше, тем лучше.");
            AddMetric(rows, root, "precision", "Precision", "Доля правильных срабатываний среди всех найденных моделью областей. Высокое значение означает меньше ложных срабатываний.");
            AddMetric(rows, root, "recall", "Recall", "Доля реально существующих областей даты, которые модель смогла найти. Высокое значение означает меньше пропусков.");
            AddMetric(rows, root, "fitness", "Fitness", "Сводная служебная метрика Ultralytics для сравнения запусков обучения.");

            foreach (var property in root.EnumerateObject())
            {
                if (rows.Any(x => x.RawKey == property.Name))
                    continue;

                rows.Add(new MetricRow
                {
                    RawKey = property.Name,
                    Name = property.Name,
                    Value = FormatMetricValue(property.Value),
                    Description = "Дополнительная метрика из JSON, который вернул клиент обучения."
                });
            }

            return rows.Count > 0
                ? rows
                : [new MetricRow { Name = "Метрики не распознаны", Value = "-", Description = "JSON метрик есть, но ожидаемые значения в нём не найдены." }];
        }
        catch
        {
            return
            [
                new MetricRow
                {
                    Name = "Metrics JSON",
                    Value = metricsJson,
                    Description = "Не удалось разобрать JSON автоматически, поэтому показано исходное значение."
                }
            ];
        }
    }

    private static void AddMetric(List<MetricRow> rows, JsonElement root, string key, string displayName, string description)
    {
        if (!root.TryGetProperty(key, out var value))
            return;

        rows.Add(new MetricRow
        {
            RawKey = key,
            Name = displayName,
            Value = FormatMetricValue(value),
            Description = description
        });
    }

    private static string FormatMetricValue(JsonElement value)
    {
        if (value.ValueKind == JsonValueKind.Number && value.TryGetDouble(out var number))
            return number >= 0 && number <= 1
                ? $"{number:P2} ({number:F4})"
                : number.ToString("0.####", CultureInfo.InvariantCulture);

        return value.ToString();
    }

    private static string TranslateStatus(string? status)
        => status?.ToLowerInvariant() switch
        {
            "queued" => "В очереди",
            "running" => "Выполняется",
            "completed" => "Завершена успешно",
            "failed" => "Завершена с ошибкой",
            "canceled" => "Остановлена",
            _ => status ?? "-"
        };

    private static string BoolText(bool? value)
        => value.HasValue ? (value.Value ? "да" : "нет") : "-";

    private static string ToLocalString(DateTime value)
        => value.ToLocalTime().ToString("dd.MM.yyyy HH:mm:ss", CultureInfo.InvariantCulture);

    private static string ToLocalString(DateTime? value)
        => value?.ToLocalTime().ToString("dd.MM.yyyy HH:mm:ss", CultureInfo.InvariantCulture) ?? "-";

    private static Label CreateSectionLabel(string text)
        => new()
        {
            Text = text,
            Dock = DockStyle.Fill,
            AutoSize = true,
            Font = new Font(SystemFonts.DefaultFont, FontStyle.Bold),
            Padding = new Padding(0, 8, 0, 8)
        };

    private static Label CreateValueLabel()
        => new()
        {
            AutoSize = true,
            Anchor = AnchorStyles.Left,
            MaximumSize = new Size(430, 0)
        };

    private static void AddInfoRow(TableLayoutPanel layout, int row, string leftTitle, Control leftValue, string rightTitle, Control rightValue)
    {
        layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));

        layout.Controls.Add(CreateTitleLabel(leftTitle), 0, row);
        leftValue.Margin = new Padding(0, 8, 12, 8);
        layout.Controls.Add(leftValue, 1, row);

        layout.Controls.Add(CreateTitleLabel(rightTitle), 2, row);
        rightValue.Margin = new Padding(0, 8, 0, 8);
        layout.Controls.Add(rightValue, 3, row);
    }

    private static Label CreateTitleLabel(string text)
        => new()
        {
            Text = text,
            AutoSize = true,
            Anchor = AnchorStyles.Left,
            Font = new Font(SystemFonts.DefaultFont, FontStyle.Bold),
            Margin = new Padding(0, 8, 8, 8)
        };

    private sealed class MetricRow
    {
        public string RawKey { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string Value { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
    }
}
