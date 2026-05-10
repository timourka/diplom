using Contracts.Dtos;
using System.Globalization;
using System.Text.Json;

namespace AdminApp;

public sealed class FormTrainingJobDetails : Form
{
    private readonly AdminApiClient _api;
    private readonly string _jobId;
    private TrainingJobStatusResponse? _job;

    private readonly Label _lblJobIdValue = CreateValueLabel();
    private readonly Label _lblStatusValue = CreateValueLabel();
    private readonly Label _lblCreatedAtValue = CreateValueLabel();
    private readonly Label _lblStartedAtValue = CreateValueLabel();
    private readonly Label _lblFinishedAtValue = CreateValueLabel();
    private readonly Label _lblAssignedAtValue = CreateValueLabel();
    private readonly Label _lblHeartbeatAtValue = CreateValueLabel();
    private readonly Label _lblImagesCountValue = CreateValueLabel();
    private readonly Label _lblBaseModelValue = CreateValueLabel();
    private readonly Label _lblMobileFormatValue = CreateValueLabel();
    private readonly Label _lblClientIdValue = CreateValueLabel();
    private readonly Label _lblParamsValue = CreateValueLabel();
    private readonly Label _lblMobileExportValue = CreateValueLabel();
    private readonly Label _lblCancellationValue = CreateValueLabel();
    private readonly TextBox _txtArtifacts = new() { Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical, Dock = DockStyle.Fill, Height = 80 };
    private readonly TextBox _txtMessage = new() { Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical, Dock = DockStyle.Fill, Height = 130 };
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
    private readonly Button _btnRefresh = new() { Text = "Обновить", Width = 120, Height = 36 };
    private readonly Button _btnStop = new() { Text = "Остановить задачу", Width = 180, Height = 36 };
    private readonly Button _btnOpenModel = new() { Text = "Открыть версию модели", Width = 210, Height = 36 };
    private readonly Button _btnClose = new() { Text = "Закрыть", Width = 120, Height = 36 };

    public FormTrainingJobDetails(AdminApiClient api, string jobId)
    {
        _api = api;
        _jobId = jobId;

        Text = $"Задача обучения: {jobId}";
        Width = 1100;
        Height = 820;
        StartPosition = FormStartPosition.CenterParent;

        BuildUi();

        _btnRefresh.Click += async (_, _) => await LoadJobAsync();
        _btnStop.Click += async (_, _) => await CancelJobAsync();
        _btnOpenModel.Click += async (_, _) => await OpenModelVersionAsync();
        _btnClose.Click += (_, _) => Close();
        Shown += async (_, _) => await LoadJobAsync();
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

        header.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 160));
        header.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
        header.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 160));
        header.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));

        AddInfoRow(header, 0, "Job ID", _lblJobIdValue, "Статус", _lblStatusValue);
        AddInfoRow(header, 1, "Создано", _lblCreatedAtValue, "Назначено клиенту", _lblAssignedAtValue);
        AddInfoRow(header, 2, "Запущено", _lblStartedAtValue, "Последний heartbeat", _lblHeartbeatAtValue);
        AddInfoRow(header, 3, "Завершено", _lblFinishedAtValue, "Кадров в датасете", _lblImagesCountValue);
        AddInfoRow(header, 4, "Клиент обучения", _lblClientIdValue, "Остановка", _lblCancellationValue);
        AddInfoRow(header, 5, "Базовая модель", _lblBaseModelValue, "Mobile формат", _lblMobileFormatValue);
        AddInfoRow(header, 6, "Параметры", _lblParamsValue, "Mobile export", _lblMobileExportValue);

        var actionsPanel = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            AutoSize = true,
            Padding = new Padding(12, 0, 12, 12),
            FlowDirection = FlowDirection.LeftToRight,
        };
        actionsPanel.Controls.AddRange([_btnRefresh, _btnStop, _btnOpenModel, _btnClose]);

        var content = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            Padding = new Padding(12, 0, 12, 12),
            ColumnCount = 1,
            RowCount = 6,
        };
        content.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        content.RowStyles.Add(new RowStyle(SizeType.Percent, 55));
        content.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        content.RowStyles.Add(new RowStyle(SizeType.Absolute, 100));
        content.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        content.RowStyles.Add(new RowStyle(SizeType.Percent, 45));

        content.Controls.Add(CreateSectionLabel("Метрики качества"), 0, 0);
        content.Controls.Add(_gridMetrics, 0, 1);
        content.Controls.Add(CreateSectionLabel("Файлы и артефакты"), 0, 2);
        content.Controls.Add(_txtArtifacts, 0, 3);
        content.Controls.Add(CreateSectionLabel("Сообщение сервиса"), 0, 4);
        content.Controls.Add(_txtMessage, 0, 5);

        Controls.Add(content);
        Controls.Add(actionsPanel);
        Controls.Add(header);
    }

    private void ConfigureMetricsGrid()
    {
        _gridMetrics.Columns.Add(new DataGridViewTextBoxColumn
        {
            HeaderText = "Метрика",
            DataPropertyName = nameof(MetricRow.Name),
            Width = 180
        });
        _gridMetrics.Columns.Add(new DataGridViewTextBoxColumn
        {
            HeaderText = "Значение",
            DataPropertyName = nameof(MetricRow.Value),
            Width = 120
        });
        _gridMetrics.Columns.Add(new DataGridViewTextBoxColumn
        {
            HeaderText = "Пояснение",
            DataPropertyName = nameof(MetricRow.Description),
            AutoSizeMode = DataGridViewAutoSizeColumnMode.Fill
        });
    }

    private async Task LoadJobAsync()
    {
        try
        {
            ToggleBusy(true);
            var job = await _api.GetTrainingJobAsync(_jobId);
            if (job is null)
            {
                MessageBox.Show(this, "Задача не найдена на сервере.", "Обучение", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                Close();
                return;
            }

            Render(job);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка загрузки задачи", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            ToggleBusy(false);
        }
    }

    private async Task CancelJobAsync()
    {
        var confirm = MessageBox.Show(
            this,
            "Остановить выбранную задачу обучения? Запущенный процесс будет прерван.",
            "Подтверждение остановки",
            MessageBoxButtons.YesNo,
            MessageBoxIcon.Question);

        if (confirm != DialogResult.Yes)
            return;

        try
        {
            ToggleBusy(true);
            var job = await _api.CancelTrainingJobAsync(_jobId);
            Render(job);
            MessageBox.Show(this, "Команда остановки отправлена.", "Обучение", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка остановки", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            ToggleBusy(false);
        }
    }

    private void Render(TrainingJobStatusResponse job)
    {
        _job = job;
        _lblJobIdValue.Text = job.JobId;
        _lblStatusValue.Text = TranslateStatus(job.Status);
        _lblCreatedAtValue.Text = ToLocalString(job.CreatedAt);
        _lblStartedAtValue.Text = ToLocalString(job.StartedAt);
        _lblFinishedAtValue.Text = ToLocalString(job.FinishedAt);
        _lblAssignedAtValue.Text = ToLocalString(job.AssignedAt);
        _lblHeartbeatAtValue.Text = ToLocalString(job.HeartbeatAt);
        _lblImagesCountValue.Text = job.ImagesCount.ToString(CultureInfo.InvariantCulture);
        _lblBaseModelValue.Text = job.BaseModel ?? "-";
        _lblMobileFormatValue.Text = job.MobileFormat ?? "-";
        _lblClientIdValue.Text = job.ClientId ?? "-";
        _lblCancellationValue.Text = job.CancellationRequested ? "запрошена" : "нет";
        _lblParamsValue.Text = BuildTrainingParams(job);
        _lblMobileExportValue.Text = BuildMobileExport(job);

        _txtArtifacts.Text = string.Join(Environment.NewLine, new[]
        {
            $"Dataset ZIP: {job.DatasetZipPath ?? "-"}",
            $"Файл весов лучшей модели: {job.BestWeightsPath ?? "-"}",
            $"Файл мобильной модели: {job.MobileModelPath ?? "-"}",
            $"Имя mobile-файла: {job.MobileModelFileName ?? "-"}",
            $"Mobile Content-Type: {job.MobileModelContentType ?? "-"}"
        });

        _txtMessage.Text = job.Message ?? "-";
        _gridMetrics.DataSource = BuildMetricRows(job.MetricsJson);
        _btnStop.Enabled = CanBeCanceled(job.Status);
        _btnOpenModel.Enabled = HasPossibleModel(job);
    }

    private async Task OpenModelVersionAsync()
    {
        if (_job is null)
            return;

        try
        {
            ToggleBusy(true);
            var versions = await _api.GetModelVersionsAsync();
            var model = versions.FirstOrDefault(x => string.Equals(x.ExternalJobId, _job.JobId, StringComparison.OrdinalIgnoreCase));
            if (model is null)
            {
                MessageBox.Show(this, "Для этой задачи пока нет сохранённой версии модели.", "Версия модели", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            using var form = new FormModelVersionDetails(_api, model);
            form.ShowDialog(this);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка открытия версии модели", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            ToggleBusy(false);
        }
    }

    private void ToggleBusy(bool busy)
    {
        UseWaitCursor = busy;
        _btnRefresh.Enabled = !busy;
        _btnClose.Enabled = !busy;
        if (busy)
        {
            _btnStop.Enabled = false;
            _btnOpenModel.Enabled = false;
        }
        else
        {
            _btnOpenModel.Enabled = _job is not null && HasPossibleModel(_job);
        }
    }

    private static List<MetricRow> BuildMetricRows(string? metricsJson)
    {
        if (string.IsNullOrWhiteSpace(metricsJson))
        {
            return
            [
                new MetricRow
                {
                    Name = "Метрики пока недоступны",
                    Value = "-",
                    Description = "Обычно метрики появляются после этапа валидации. Пока обучение ещё не дошло до оценки качества или задача завершилась до расчёта метрик."
                }
            ];
        }

        try
        {
            using var document = JsonDocument.Parse(metricsJson);
            var root = document.RootElement;
            var rows = new List<MetricRow>();

            AddMetric(rows, root, "mAP50_95", "mAP@0.50:0.95", "Главная итоговая метрика детектора. Показывает среднее качество обнаружения на нескольких уровнях строгости IoU от 0.50 до 0.95. Чем выше, тем лучше.");
            AddMetric(rows, root, "mAP50", "mAP@0.50", "Качество обнаружения при более мягком критерии совпадения рамок. Обычно заметно выше, чем mAP@0.50:0.95. Чем выше, тем лучше.");
            AddMetric(rows, root, "mAP75", "mAP@0.75", "Качество обнаружения при более строгом сравнении рамок. Показывает, насколько точно модель позиционирует область даты. Чем выше, тем лучше.");
            AddMetric(rows, root, "precision", "Precision", "Доля правильных срабатываний среди всех найденных моделью областей. Высокое значение означает меньше ложных срабатываний.");
            AddMetric(rows, root, "recall", "Recall", "Доля реально существующих областей даты, которые модель смогла найти. Высокое значение означает меньше пропусков.");
            AddMetric(rows, root, "fitness", "Fitness", "Сводная служебная метрика Ultralytics для сравнения запусков обучения. Используется как интегральная оценка качества модели.");

            foreach (var property in root.EnumerateObject())
            {
                if (rows.Any(x => x.RawKey == property.Name))
                    continue;

                rows.Add(new MetricRow
                {
                    RawKey = property.Name,
                    Name = property.Name,
                    Value = FormatMetricValue(property.Value),
                    Description = "Дополнительная метрика, сохранённая сервисом обучения."
                });
            }

            return rows.Count > 0
                ? rows
                : [new MetricRow { Name = "Метрики не распознаны", Value = "-", Description = "Сервис вернул JSON метрик, но он не содержал ожидаемых числовых значений." }];
        }
        catch
        {
            return
            [
                new MetricRow
                {
                    Name = "Метрики",
                    Value = metricsJson,
                    Description = "Не удалось разобрать JSON автоматически, поэтому показано исходное значение, которое вернул сервис обучения."
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
            return $"{number:P2} ({number:F4})";

        return value.ToString();
    }

    private static string BuildTrainingParams(TrainingJobStatusResponse job)
        => $"epochs={job.Epochs?.ToString(CultureInfo.InvariantCulture) ?? "-"}, "
           + $"imgsz={job.ImgSize?.ToString(CultureInfo.InvariantCulture) ?? "-"}, "
           + $"batch={job.Batch?.ToString(CultureInfo.InvariantCulture) ?? "-"}, "
           + $"device={job.Device ?? "-"}";

    private static string BuildMobileExport(TrainingJobStatusResponse job)
        => $"int8={BoolText(job.ExportInt8)}, "
           + $"nms={BoolText(job.ExportNms)}, "
           + $"quant={job.QuantizationFraction?.ToString("0.###", CultureInfo.InvariantCulture) ?? "-"}";

    private static string BoolText(bool? value)
        => value.HasValue ? (value.Value ? "да" : "нет") : "-";

    private static bool HasPossibleModel(TrainingJobStatusResponse job)
        => string.Equals(job.Status, "completed", StringComparison.OrdinalIgnoreCase)
           || !string.IsNullOrWhiteSpace(job.MobileModelPath)
           || !string.IsNullOrWhiteSpace(job.BestWeightsPath);

    private static bool CanBeCanceled(string? status)
        => string.Equals(status, "queued", StringComparison.OrdinalIgnoreCase)
           || string.Equals(status, "running", StringComparison.OrdinalIgnoreCase);

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

    private static string ToLocalString(DateTime? value)
        => value?.ToLocalTime().ToString("dd.MM.yyyy HH:mm:ss") ?? "-";

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
            MaximumSize = new Size(420, 0)
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
