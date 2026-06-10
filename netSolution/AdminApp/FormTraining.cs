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
    private readonly Button _btnDetails = new() { Text = "Подробнее по задаче", Width = 180, Height = 36 };
    private readonly Button _btnStop = new() { Text = "Остановить задачу", Width = 170, Height = 36 };
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
    private readonly CancellationTokenSource _lifetimeCts = new();
    private bool _isClosing;
    private bool _isRefreshing;
    private bool _isLoadingSelectedJob;

    private bool CanTouchUi => !_isClosing && !Disposing && !IsDisposed && IsHandleCreated;

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
        _btnDetails.Click += async (_, _) => await OpenDetailsAsync();
        _btnStop.Click += async (_, _) => await CancelSelectedJobAsync();
        _gridJobs.SelectionChanged += async (_, _) => await ShowSelectedJobAsync();
        _gridJobs.CellDoubleClick += async (_, _) => await OpenDetailsAsync();
        _timer.Tick += async (_, _) => await RefreshJobsAsync(silent: true);
        Shown += async (_, _) => await RefreshJobsAsync();
        FormClosing += (_, _) =>
        {
            _isClosing = true;
            _timer.Stop();
            _lifetimeCts.Cancel();
        };
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
        buttons.Controls.Add(_btnDetails);
        buttons.Controls.Add(_btnStop);

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
        _gridJobs.Columns.Add(new DataGridViewTextBoxColumn { Name = "Status", HeaderText = "Статус", DataPropertyName = "Status", Width = 140 });
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
            AppendStatus("Создаю задачу обучения на backend...");

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
            ), _lifetimeCts.Token);

            if (!CanTouchUi)
                return;

            _txtJobId.Text = response.JobId;
            AppendStatus($"Job {response.JobId}: {TranslateStatus(response.Status)}. {response.Message}");
            await RefreshJobsAsync(selectJobId: response.JobId);

            if (CanTouchUi)
                _timer.Start();
        }
        catch (OperationCanceledException) when (_lifetimeCts.IsCancellationRequested)
        {
            // Форма закрывается: не показываем ошибку и не трогаем уже уничтоженные контролы.
        }
        catch (ObjectDisposedException) when (!CanTouchUi)
        {
            // Асинхронный запрос завершился уже после закрытия окна. Это не ошибка пользователя.
        }
        catch (Exception ex)
        {
            ShowErrorSafe(ex.Message, "Ошибка обучения");
            AppendStatus("Ошибка: " + ex.Message);
        }
        finally
        {
            if (CanTouchUi)
                ToggleBusy(false);
        }
    }

    private async Task RefreshJobsAsync(bool silent = false, string? selectJobId = null)
    {
        if (!CanTouchUi || _isRefreshing)
            return;

        _isRefreshing = true;
        try
        {
            var jobs = await _api.GetTrainingJobsAsync(_lifetimeCts.Token);

            if (!CanTouchUi)
                return;

            var selectedJobId = selectJobId ?? GetSelectedJobId() ?? _txtJobId.Text.Trim();

            _gridJobs.DataSource = jobs
                .Select(x => new TrainingJobGridRow
                {
                    JobId = x.JobId,
                    Status = TranslateStatus(x.Status),
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
            UpdateButtonsState();

            if (!silent)
                AppendStatus($"Список задач обучения обновлён. Всего: {jobs.Count}.");
        }
        catch (OperationCanceledException) when (_lifetimeCts.IsCancellationRequested)
        {
            // Окно закрывается.
        }
        catch (ObjectDisposedException) when (!CanTouchUi)
        {
            // Асинхронное обновление завершилось после закрытия окна.
        }
        catch (Exception ex)
        {
            if (!silent)
                AppendStatus("Ошибка обновления списка: " + ex.Message);
        }
        finally
        {
            _isRefreshing = false;
        }
    }

    private async Task ShowSelectedJobAsync()
    {
        if (!CanTouchUi || _isLoadingSelectedJob)
            return;

        var selected = GetSelectedJob();
        UpdateButtonsState();
        if (selected is null)
            return;

        _txtJobId.Text = selected.JobId;

        _isLoadingSelectedJob = true;
        try
        {
            var job = await _api.GetTrainingJobAsync(selected.JobId, _lifetimeCts.Token);

            if (!CanTouchUi)
                return;

            if (job is null)
            {
                _txtStatus.Text = "Задача не найдена на backend.";
                return;
            }

            var lines = new List<string>
            {
                $"Job ID: {job.JobId}",
                $"Статус: {TranslateStatus(job.Status)}",
                $"Создано: {ToLocalString(job.CreatedAt)}",
                $"Старт: {ToLocalString(job.StartedAt)}",
                $"Финиш: {ToLocalString(job.FinishedAt)}",
                $"Кадров в датасете: {job.ImagesCount}",
                $"Базовая модель: {job.BaseModel}",
                $"Mobile формат: {job.MobileFormat}",
                $"Файл лучших весов: {job.BestWeightsPath ?? "-"}",
                $"Файл мобильной модели: {job.MobileModelPath ?? "-"}",
                "",
                "Кратко по метрикам:",
                BuildMetricsSummary(job.MetricsJson),
                "",
                "Сообщение сервиса:",
                job.Message ?? "-",
                "",
                "Подсказка: дважды щёлкните по задаче в списке выше, чтобы открыть подробное окно с пояснениями по каждой метрике. Готовые модели появятся в окне «Версии модели»."
            };

            _txtStatus.Text = string.Join(Environment.NewLine, lines);

            if (IsTerminal(job.Status))
            {
                var rows = (_gridJobs.DataSource as List<TrainingJobGridRow>) ?? [];
                if (rows.All(x => IsTerminal(x.Source.Status)))
                    _timer.Stop();
            }
        }
        catch (OperationCanceledException) when (_lifetimeCts.IsCancellationRequested)
        {
            // Окно закрывается.
        }
        catch (ObjectDisposedException) when (!CanTouchUi)
        {
            // Асинхронная загрузка завершилась после закрытия окна.
        }
        catch (Exception ex)
        {
            if (CanTouchUi)
                _txtStatus.Text = "Ошибка получения задачи: " + ex.Message;
        }
        finally
        {
            _isLoadingSelectedJob = false;
        }
    }

    private async Task OpenDetailsAsync()
    {
        var selected = GetSelectedJob();
        if (selected is null)
        {
            MessageBox.Show(this, "Сначала выберите задачу обучения в списке.", "Обучение", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        using var dialog = new FormTrainingJobDetails(_api, selected.JobId);
        dialog.ShowDialog(this);

        if (CanTouchUi)
            await RefreshJobsAsync(selectJobId: selected.JobId);
    }

    private async Task CancelSelectedJobAsync()
    {
        var selected = GetSelectedJob();
        if (selected is null)
        {
            MessageBox.Show(this, "Сначала выберите задачу обучения в списке.", "Обучение", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        if (!CanBeCanceled(selected.Source.Status))
        {
            MessageBox.Show(this, "Эту задачу уже нельзя остановить: она завершена, остановлена или завершилась с ошибкой.", "Обучение", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        var confirm = MessageBox.Show(
            this,
            $"Остановить задачу {selected.JobId}?",
            "Подтверждение остановки",
            MessageBoxButtons.YesNo,
            MessageBoxIcon.Question);

        if (confirm != DialogResult.Yes)
            return;

        try
        {
            ToggleBusy(true);
            var response = await _api.CancelTrainingJobAsync(selected.JobId, _lifetimeCts.Token);

            if (!CanTouchUi)
                return;
            AppendStatus($"Задача {selected.JobId}: {TranslateStatus(response.Status)}. {response.Message}");
            await RefreshJobsAsync(selectJobId: selected.JobId);
        }
        catch (OperationCanceledException) when (_lifetimeCts.IsCancellationRequested)
        {
            // Окно закрывается.
        }
        catch (ObjectDisposedException) when (!CanTouchUi)
        {
            // Асинхронный запрос завершился после закрытия окна.
        }
        catch (Exception ex)
        {
            ShowErrorSafe(ex.Message, "Ошибка остановки");
            AppendStatus("Ошибка остановки задачи: " + ex.Message);
        }
        finally
        {
            if (CanTouchUi)
                ToggleBusy(false);
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

    private void UpdateButtonsState()
    {
        var selected = GetSelectedJob();
        _btnDetails.Enabled = selected is not null;
        _btnStop.Enabled = selected is not null && CanBeCanceled(selected.Source.Status);
    }

    private static bool IsTerminal(string? status)
        => string.Equals(status, "completed", StringComparison.OrdinalIgnoreCase)
           || string.Equals(status, "failed", StringComparison.OrdinalIgnoreCase)
           || string.Equals(status, "canceled", StringComparison.OrdinalIgnoreCase);

    private static bool CanBeCanceled(string? status)
        => string.Equals(status, "preparing", StringComparison.OrdinalIgnoreCase)
           || string.Equals(status, "queued", StringComparison.OrdinalIgnoreCase)
           || string.Equals(status, "running", StringComparison.OrdinalIgnoreCase);

    private static string TranslateStatus(string? status)
        => status?.ToLowerInvariant() switch
        {
            "preparing" => "Подготовка датасета",
            "queued" => "В очереди",
            "running" => "Выполняется",
            "completed" => "Завершена успешно",
            "failed" => "Завершена с ошибкой",
            "canceled" => "Остановлена",
            _ => status ?? "-"
        };

    private static string ToLocalString(DateTime? value)
        => value?.ToLocalTime().ToString("dd.MM.yyyy HH:mm:ss") ?? "-";

    private static string BuildMetricsSummary(string? metricsJson)
    {
        if (string.IsNullOrWhiteSpace(metricsJson))
            return "Метрики пока не рассчитаны.";

        try
        {
            using var document = JsonDocument.Parse(metricsJson);
            var root = document.RootElement;
            var lines = new List<string>();

            AppendMetric(lines, root, "mAP50_95", "mAP@0.50:0.95 — основная итоговая метрика качества детектора");
            AppendMetric(lines, root, "mAP50", "mAP@0.50 — качество при мягком критерии совпадения рамок");
            AppendMetric(lines, root, "mAP75", "mAP@0.75 — качество при более строгом совпадении рамок");
            AppendMetric(lines, root, "precision", "Precision — доля правильных срабатываний среди всех найденных");
            AppendMetric(lines, root, "recall", "Recall — доля реальных дат, которые модель нашла");
            AppendMetric(lines, root, "fitness", "Fitness — сводная служебная метрика Ultralytics");

            return lines.Count > 0 ? string.Join(Environment.NewLine, lines) : PrettyJson(metricsJson);
        }
        catch
        {
            return PrettyJson(metricsJson);
        }
    }

    private static void AppendMetric(List<string> lines, JsonElement root, string key, string title)
    {
        if (!root.TryGetProperty(key, out var value) || !value.TryGetDouble(out var number))
            return;

        lines.Add($"• {title}: {number:P2} ({number:F4})");
    }

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

    private void ShowErrorSafe(string message, string title)
    {
        if (!CanTouchUi)
            return;

        try
        {
            MessageBox.Show(this, message, title, MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        catch (ObjectDisposedException)
        {
            // Окно уже закрыто.
        }
    }

    private void AppendStatus(string text)
    {
        if (!CanTouchUi)
            return;

        if (InvokeRequired)
        {
            try
            {
                BeginInvoke(new Action(() => AppendStatus(text)));
            }
            catch
            {
                // Окно закрывается.
            }

            return;
        }

        try
        {
            _txtStatus.AppendText(text + Environment.NewLine);
        }
        catch (ObjectDisposedException)
        {
            // Окно уже закрыто.
        }
    }

    private void ToggleBusy(bool busy)
    {
        if (!CanTouchUi)
            return;

        UseWaitCursor = busy;
        _btnStart.Enabled = !busy;
        _btnRefresh.Enabled = !busy;
        if (busy)
        {
            _btnDetails.Enabled = false;
            _btnStop.Enabled = false;
        }
        else
        {
            UpdateButtonsState();
        }
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
