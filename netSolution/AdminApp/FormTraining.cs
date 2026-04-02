using Contracts.Dtos;
using System.Text.Json;

namespace AdminApp;

public class FormTraining : Form
{
    private readonly AdminApiClient _api;
    private readonly TextBox _txtBaseModel = new() { Text = "yolov8n.pt", Width = 180 };
    private readonly NumericUpDown _numEpochs = new() { Minimum = 1, Maximum = 1000, Value = 50 };
    private readonly NumericUpDown _numImgsz = new() { Minimum = 320, Maximum = 1280, Increment = 32, Value = 640 };
    private readonly NumericUpDown _numBatch = new() { Minimum = 1, Maximum = 128, Value = 16 };
    private readonly TextBox _txtDevice = new() { Text = "auto", Width = 80 };
    private readonly CheckBox _chkInt8 = new() { Text = "INT8 для mobile", Checked = true, AutoSize = true };
    private readonly CheckBox _chkNms = new() { Text = "Добавить NMS", Checked = true, AutoSize = true };
    private readonly TextBox _txtJobId = new() { ReadOnly = true, Width = 360 };
    private readonly TextBox _txtStatus = new() { Multiline = true, ScrollBars = ScrollBars.Vertical, Dock = DockStyle.Fill, ReadOnly = true };
    private readonly Button _btnStart = new() { Text = "Запустить обучение", Width = 180, Height = 36 };
    private readonly Button _btnRefresh = new() { Text = "Обновить список", Width = 140, Height = 36 };
    private readonly DataGridView _gridJobs = new()
    {
        Dock = DockStyle.Fill,
        ReadOnly = true,
        AllowUserToAddRows = false,
        AllowUserToDeleteRows = false,
        AutoGenerateColumns = false,
        SelectionMode = DataGridViewSelectionMode.FullRowSelect,
        MultiSelect = false,
        RowHeadersVisible = false,
    };
    private readonly System.Windows.Forms.Timer _timer = new() { Interval = 4000 };

    public FormTraining(AdminApiClient api)
    {
        _api = api;

        Text = "Обучение модели";
        Width = 1200;
        Height = 700;
        StartPosition = FormStartPosition.CenterParent;

        BuildUi();

        _btnStart.Click += async (_, _) => await StartAsync();
        _btnRefresh.Click += async (_, _) => await RefreshJobsAsync();
        _gridJobs.SelectionChanged += async (_, _) => await ShowSelectedJobAsync();
        _timer.Tick += async (_, _) => await RefreshJobsAsync(silent: true);
        Shown += async (_, _) => await RefreshJobsAsync();
        FormClosed += (_, _) => _timer.Stop();
    }

    private void BuildUi()
    {
        var topLayout = new TableLayoutPanel
        {
            Dock = DockStyle.Top,
            ColumnCount = 4,
            AutoSize = true,
            Padding = new Padding(12),
        };

        topLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 160));
        topLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 220));
        topLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 160));
        topLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

        AddRow(topLayout, 0, "Базовая модель", _txtBaseModel, "Epochs", _numEpochs);
        AddRow(topLayout, 1, "Размер изображения", _numImgsz, "Batch", _numBatch);
        AddRow(topLayout, 2, "Устройство", _txtDevice, "Job ID", _txtJobId);

        var checksPanel = new FlowLayoutPanel { AutoSize = true, FlowDirection = FlowDirection.LeftToRight };
        checksPanel.Controls.Add(_chkInt8);
        checksPanel.Controls.Add(_chkNms);

        var buttons = new FlowLayoutPanel { AutoSize = true, FlowDirection = FlowDirection.LeftToRight };
        buttons.Controls.Add(_btnStart);
        buttons.Controls.Add(_btnRefresh);

        AddRow(topLayout, 3, "Mobile export", checksPanel, "Действия", buttons);

        ConfigureGrid();

        var split = new SplitContainer
        {
            Dock = DockStyle.Fill,
            Orientation = Orientation.Horizontal,
            SplitterDistance = 320,
        };

        split.Panel1.Controls.Add(_gridJobs);
        split.Panel2.Controls.Add(_txtStatus);

        Controls.Add(split);
        Controls.Add(topLayout);
    }

    private void ConfigureGrid()
    {
        _gridJobs.Columns.Add(new DataGridViewTextBoxColumn { Name = "JobId", HeaderText = "Job ID", DataPropertyName = "JobId", Width = 240 });
        _gridJobs.Columns.Add(new DataGridViewTextBoxColumn { Name = "Status", HeaderText = "Статус", DataPropertyName = "Status", Width = 90 });
        _gridJobs.Columns.Add(new DataGridViewTextBoxColumn { Name = "CreatedAt", HeaderText = "Создано", DataPropertyName = "CreatedAt", Width = 150 });
        _gridJobs.Columns.Add(new DataGridViewTextBoxColumn { Name = "StartedAt", HeaderText = "Старт", DataPropertyName = "StartedAt", Width = 150 });
        _gridJobs.Columns.Add(new DataGridViewTextBoxColumn { Name = "FinishedAt", HeaderText = "Финиш", DataPropertyName = "FinishedAt", Width = 150 });
        _gridJobs.Columns.Add(new DataGridViewTextBoxColumn { Name = "ImagesCount", HeaderText = "Кадров", DataPropertyName = "ImagesCount", Width = 80 });
        _gridJobs.Columns.Add(new DataGridViewTextBoxColumn { Name = "BaseModel", HeaderText = "Модель", DataPropertyName = "BaseModel", Width = 120 });
        _gridJobs.Columns.Add(new DataGridViewTextBoxColumn { Name = "MobileFormat", HeaderText = "Mobile", DataPropertyName = "MobileFormat", Width = 80 });
        _gridJobs.Columns.Add(new DataGridViewTextBoxColumn { Name = "Message", HeaderText = "Сообщение", DataPropertyName = "Message", AutoSizeMode = DataGridViewAutoSizeColumnMode.Fill });
    }

    private static void AddRow(TableLayoutPanel layout, int row, string labelLeft, Control controlLeft, string labelRight, Control controlRight)
    {
        layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));

        layout.Controls.Add(new Label
        {
            Text = labelLeft,
            AutoSize = true,
            Anchor = AnchorStyles.Left,
            Margin = new Padding(0, 8, 8, 8)
        }, 0, row);
        controlLeft.Anchor = AnchorStyles.Left | AnchorStyles.Right;
        controlLeft.Margin = new Padding(0, 4, 16, 4);
        layout.Controls.Add(controlLeft, 1, row);

        layout.Controls.Add(new Label
        {
            Text = labelRight,
            AutoSize = true,
            Anchor = AnchorStyles.Left,
            Margin = new Padding(0, 8, 8, 8)
        }, 2, row);
        controlRight.Anchor = AnchorStyles.Left | AnchorStyles.Right;
        controlRight.Margin = new Padding(0, 4, 0, 4);
        layout.Controls.Add(controlRight, 3, row);
    }

    private async Task StartAsync()
    {
        try
        {
            ToggleBusy(true);
            AppendStatus("Запускаю обучение и отправку approved-датасета на training-service...");

            var response = await _api.StartTrainingAsync(new StartTrainingRequest(
                _txtBaseModel.Text.Trim(),
                (int)_numEpochs.Value,
                (int)_numImgsz.Value,
                (int)_numBatch.Value,
                _txtDevice.Text.Trim(),
                _chkInt8.Checked,
                _chkNms.Checked,
                "tflite",
                0.3
            ));

            _txtJobId.Text = response.JobId;
            AppendStatus($"Job {response.JobId}: {response.Status}. {response.Message}");
            await RefreshJobsAsync(selectJobId: response.JobId);
            _timer.Start();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка обучения", MessageBoxButtons.OK, MessageBoxIcon.Error);
            AppendStatus("Ошибка: " + ex.Message);
        }
        finally
        {
            ToggleBusy(false);
        }
    }

    private async Task RefreshJobsAsync(bool silent = false, string? selectJobId = null)
    {
        try
        {
            var jobs = await _api.GetTrainingJobsAsync();
            var selectedJobId = selectJobId ?? GetSelectedJobId() ?? _txtJobId.Text.Trim();

            _gridJobs.DataSource = jobs
                .Select(x => new TrainingJobGridRow
                {
                    JobId = x.JobId,
                    Status = x.Status,
                    CreatedAt = ToLocalString(x.CreatedAt),
                    StartedAt = ToLocalString(x.StartedAt),
                    FinishedAt = ToLocalString(x.FinishedAt),
                    ImagesCount = x.ImagesCount,
                    BaseModel = x.BaseModel,
                    MobileFormat = x.MobileFormat,
                    Message = x.Message,
                    Source = x,
                })
                .ToList();

            RestoreSelection(selectedJobId);

            if (!silent)
                AppendStatus($"Список job обновлён. Всего: {jobs.Count}.");
        }
        catch (Exception ex)
        {
            if (!silent)
                AppendStatus("Ошибка обновления списка: " + ex.Message);
        }
    }

    private async Task ShowSelectedJobAsync()
    {
        var selected = GetSelectedJob();
        if (selected is null)
            return;

        _txtJobId.Text = selected.JobId;

        try
        {
            var job = await _api.GetTrainingJobAsync(selected.JobId);
            if (job is null)
            {
                _txtStatus.Text = "Job не найден на сервере.";
                return;
            }

            var lines = new List<string>
            {
                $"Job ID: {job.JobId}",
                $"Статус: {job.Status}",
                $"Создано: {ToLocalString(job.CreatedAt)}",
                $"Старт: {ToLocalString(job.StartedAt)}",
                $"Финиш: {ToLocalString(job.FinishedAt)}",
                $"Кадров: {job.ImagesCount}",
                $"Базовая модель: {job.BaseModel}",
                $"Mobile формат: {job.MobileFormat}",
                $"Best weights: {job.BestWeightsPath ?? "-"}",
                $"Mobile model: {job.MobileModelPath ?? "-"}",
                "",
                "Сообщение:",
                job.Message ?? "-"
            };

            if (!string.IsNullOrWhiteSpace(job.MetricsJson))
            {
                lines.Add("");
                lines.Add("Метрики:");
                lines.Add(PrettyJson(job.MetricsJson));
            }

            _txtStatus.Text = string.Join(Environment.NewLine, lines);

            if (IsTerminal(job.Status))
            {
                var rows = (_gridJobs.DataSource as List<TrainingJobGridRow>) ?? [];
                if (rows.All(x => IsTerminal(x.Source.Status)))
                    _timer.Stop();
            }
        }
        catch (Exception ex)
        {
            _txtStatus.Text = "Ошибка получения job: " + ex.Message;
        }
    }

    private TrainingJobGridRow? GetSelectedJob()
        => _gridJobs.CurrentRow?.DataBoundItem as TrainingJobGridRow;

    private string? GetSelectedJobId()
        => GetSelectedJob()?.JobId;

    private void RestoreSelection(string? jobId)
    {
        if (string.IsNullOrWhiteSpace(jobId))
            return;

        foreach (DataGridViewRow row in _gridJobs.Rows)
        {
            if (row.DataBoundItem is TrainingJobGridRow item && item.JobId == jobId)
            {
                row.Selected = true;
                _gridJobs.CurrentCell = row.Cells[0];
                break;
            }
        }
    }

    private static bool IsTerminal(string? status)
        => string.Equals(status, "completed", StringComparison.OrdinalIgnoreCase)
           || string.Equals(status, "failed", StringComparison.OrdinalIgnoreCase);

    private static string ToLocalString(DateTime? value)
        => value?.ToLocalTime().ToString("dd.MM.yyyy HH:mm:ss") ?? "-";

    private static string PrettyJson(string json)
    {
        try
        {
            using var document = JsonDocument.Parse(json);
            return JsonSerializer.Serialize(document.RootElement, new JsonSerializerOptions { WriteIndented = true });
        }
        catch
        {
            return json;
        }
    }

    private void AppendStatus(string text)
    {
        _txtStatus.AppendText(text + Environment.NewLine);
    }

    private void ToggleBusy(bool busy)
    {
        UseWaitCursor = busy;
        _btnStart.Enabled = !busy;
        _btnRefresh.Enabled = !busy;
    }

    private sealed class TrainingJobGridRow
    {
        public string JobId { get; set; } = string.Empty;
        public string Status { get; set; } = string.Empty;
        public string CreatedAt { get; set; } = string.Empty;
        public string StartedAt { get; set; } = string.Empty;
        public string FinishedAt { get; set; } = string.Empty;
        public int ImagesCount { get; set; }
        public string? BaseModel { get; set; }
        public string? MobileFormat { get; set; }
        public string? Message { get; set; }
        public required TrainingJobStatusResponse Source { get; set; }
    }
}
