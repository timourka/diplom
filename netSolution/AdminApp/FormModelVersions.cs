using System.Globalization;
using System.Text.Json;
using Contracts.Dtos;

namespace AdminApp;

public class FormModelVersions : Form
{
    private readonly AdminApiClient _api;
    private readonly DataGridView _grid = new()
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
    private readonly TextBox _txtDetails = new()
    {
        Dock = DockStyle.Fill,
        Multiline = true,
        ReadOnly = true,
        ScrollBars = ScrollBars.Vertical,
    };
    private readonly Button _btnRefresh = new() { Text = "Обновить", Width = 120, Height = 36 };
    private readonly Button _btnDetails = new() { Text = "Подробнее", Width = 120, Height = 36 };
    private readonly Button _btnPublish = new() { Text = "Сделать текущей", Width = 160, Height = 36 };
    private readonly Button _btnUnpublish = new() { Text = "Снять публикацию", Width = 160, Height = 36 };
    private readonly Button _btnPin = new() { Text = "Зафиксировать", Width = 140, Height = 36 };
    private readonly Button _btnUnpin = new() { Text = "Снять фиксацию", Width = 150, Height = 36 };
    private readonly Button _btnDelete = new() { Text = "Удалить", Width = 120, Height = 36 };
    private readonly Button _btnClose = new() { Text = "Закрыть", Width = 120, Height = 36 };

    public FormModelVersions(AdminApiClient api)
    {
        _api = api;
        Text = "Управление версиями модели";
        Width = 1250;
        Height = 760;
        StartPosition = FormStartPosition.CenterParent;

        BuildUi();
        _btnRefresh.Click += async (_, _) => await RefreshAsync();
        _btnDetails.Click += (_, _) => OpenDetails();
        _btnPublish.Click += async (_, _) => await PublishSelectedAsync();
        _btnUnpublish.Click += async (_, _) => await UnpublishSelectedAsync();
        _btnPin.Click += async (_, _) => await PinSelectedAsync();
        _btnUnpin.Click += async (_, _) => await UnpinSelectedAsync();
        _btnDelete.Click += async (_, _) => await DeleteSelectedAsync();
        _btnClose.Click += (_, _) => Close();
        _grid.SelectionChanged += (_, _) => RenderSelected();
        _grid.CellDoubleClick += (_, _) => OpenDetails();
        Shown += async (_, _) => await RefreshAsync();
    }

    private void BuildUi()
    {
        ConfigureGrid();

        var actions = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            AutoSize = true,
            Padding = new Padding(12),
            FlowDirection = FlowDirection.LeftToRight,
        };
        actions.Controls.AddRange([_btnRefresh, _btnDetails, _btnPublish, _btnUnpublish, _btnPin, _btnUnpin, _btnDelete, _btnClose]);

        var split = new SplitContainer
        {
            Dock = DockStyle.Fill,
            Orientation = Orientation.Horizontal,
            SplitterDistance = 420,
        };
        split.Panel1.Controls.Add(_grid);
        split.Panel2.Controls.Add(_txtDetails);

        Controls.Add(split);
        Controls.Add(actions);
    }

    private void ConfigureGrid()
    {
        _grid.Columns.Add(new DataGridViewTextBoxColumn { HeaderText = "ID", DataPropertyName = nameof(ModelVersionGridRow.Id), Width = 70 });
        _grid.Columns.Add(new DataGridViewCheckBoxColumn { HeaderText = "Текущая", DataPropertyName = nameof(ModelVersionGridRow.IsPublished), Width = 90 });
        _grid.Columns.Add(new DataGridViewCheckBoxColumn { HeaderText = "Фикс.", DataPropertyName = nameof(ModelVersionGridRow.IsPinned), Width = 70 });
        _grid.Columns.Add(new DataGridViewTextBoxColumn { HeaderText = "Обучена", DataPropertyName = nameof(ModelVersionGridRow.TrainedAt), Width = 160 });
        _grid.Columns.Add(new DataGridViewTextBoxColumn { HeaderText = "Job ID", DataPropertyName = nameof(ModelVersionGridRow.ExternalJobId), Width = 240 });
        _grid.Columns.Add(new DataGridViewTextBoxColumn { HeaderText = "База", DataPropertyName = nameof(ModelVersionGridRow.BaseModel), Width = 120 });
        _grid.Columns.Add(new DataGridViewTextBoxColumn { HeaderText = "Формат", DataPropertyName = nameof(ModelVersionGridRow.MobileFormat), Width = 80 });
        _grid.Columns.Add(new DataGridViewTextBoxColumn { HeaderText = "Файл", DataPropertyName = nameof(ModelVersionGridRow.MobileModelFileName), Width = 220 });
        _grid.Columns.Add(new DataGridViewTextBoxColumn { HeaderText = "Метрики", DataPropertyName = nameof(ModelVersionGridRow.MetricsSummary), AutoSizeMode = DataGridViewAutoSizeColumnMode.Fill });
    }

    private async Task RefreshAsync(int? selectId = null)
    {
        try
        {
            ToggleBusy(true);
            var selectedId = selectId ?? GetSelected()?.Id;
            var models = await _api.GetModelVersionsAsync();
            _grid.DataSource = models.Select(x => new ModelVersionGridRow
            {
                Id = x.Id,
                ExternalJobId = x.ExternalJobId ?? "-",
                TrainedAt = ToLocalString(x.TrainedAt),
                BaseModel = x.BaseModel ?? "-",
                MobileFormat = x.MobileFormat ?? "-",
                MobileModelFileName = x.MobileModelFileName ?? "-",
                IsPublished = x.IsPublished,
                IsPinned = x.IsPinned,
                MetricsSummary = BuildMetricsShort(x.MetricsJson),
                Source = x,
            }).ToList();
            RestoreSelection(selectedId);
            RenderSelected();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка загрузки версий", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            ToggleBusy(false);
        }
    }

    private void OpenDetails()
    {
        var selected = GetSelected();
        if (selected is null) return;

        using var form = new FormModelVersionDetails(_api, selected.Source);
        form.ShowDialog(this);
    }

    private async Task PublishSelectedAsync()
    {
        var selected = GetSelected();
        if (selected is null) return;

        var confirm = MessageBox.Show(
            this,
            $"Сделать версию #{selected.Id} текущей для скачивания мобильными пользователями?",
            "Публикация модели",
            MessageBoxButtons.YesNo,
            MessageBoxIcon.Question);
        if (confirm != DialogResult.Yes) return;

        await RunActionAsync(async () => await _api.PublishModelVersionAsync(selected.Id), selected.Id);
    }

    private async Task UnpublishSelectedAsync()
    {
        var selected = GetSelected();
        if (selected is null) return;

        await RunActionAsync(async () => await _api.UnpublishModelVersionAsync(selected.Id), selected.Id);
    }

    private async Task PinSelectedAsync()
    {
        var selected = GetSelected();
        if (selected is null) return;

        await RunActionAsync(async () => await _api.PinModelVersionAsync(selected.Id), selected.Id);
    }

    private async Task UnpinSelectedAsync()
    {
        var selected = GetSelected();
        if (selected is null) return;

        await RunActionAsync(async () => await _api.UnpinModelVersionAsync(selected.Id), selected.Id);
    }

    private async Task DeleteSelectedAsync()
    {
        var selected = GetSelected();
        if (selected is null) return;

        var text = selected.IsPinned
            ? $"Версия #{selected.Id} зафиксирована. Удалить принудительно? Мобильные пользователи больше не смогут скачать её, если она текущая."
            : $"Удалить версию #{selected.Id}? Файлы модели будут удалены с backend-хранилища.";

        var confirm = MessageBox.Show(this, text, "Удаление версии", MessageBoxButtons.YesNo, MessageBoxIcon.Warning);
        if (confirm != DialogResult.Yes) return;

        try
        {
            ToggleBusy(true);
            await _api.DeleteModelVersionAsync(selected.Id, force: selected.IsPinned);
            await RefreshAsync();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка удаления", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            ToggleBusy(false);
        }
    }

    private async Task RunActionAsync(Func<Task<ModelVersionAdminResponse>> action, int selectId)
    {
        try
        {
            ToggleBusy(true);
            await action();
            await RefreshAsync(selectId);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка управления версией", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            ToggleBusy(false);
        }
    }

    private void RenderSelected()
    {
        var selected = GetSelected();
        var hasSelected = selected is not null;
        _btnDetails.Enabled = hasSelected;
        _btnPublish.Enabled = hasSelected && !selected!.IsPublished;
        _btnUnpublish.Enabled = hasSelected && selected!.IsPublished;
        _btnPin.Enabled = hasSelected && !selected!.IsPinned;
        _btnUnpin.Enabled = hasSelected && selected!.IsPinned;
        _btnDelete.Enabled = hasSelected;

        if (selected is null)
        {
            _txtDetails.Text = "Выберите версию модели.";
            return;
        }

        var src = selected.Source;
        _txtDetails.Text = string.Join(Environment.NewLine, new[]
        {
            $"ID: {src.Id}",
            $"Job ID: {src.ExternalJobId ?? "-"}",
            $"Текущая для пользователей: {(src.IsPublished ? "Да" : "Нет")}",
            $"Зафиксирована: {(src.IsPinned ? "Да" : "Нет")}",
            $"Обучена: {ToLocalString(src.TrainedAt)}",
            $"Базовая модель: {src.BaseModel ?? "-"}",
            $"Mobile формат: {src.MobileFormat ?? "-"}",
            $"Mobile файл: {src.MobileModelFileName ?? "-"}",
            $"Mobile path на backend: {src.MobileModelPath ?? "-"}",
            $"Best weights path на backend: {src.BestWeightsPath ?? "-"}",
            "",
            "Метрики:",
            PrettyJson(src.MetricsJson),
        });
    }

    private ModelVersionGridRow? GetSelected()
        => _grid.CurrentRow?.DataBoundItem as ModelVersionGridRow;

    private void RestoreSelection(int? id)
    {
        if (id is null) return;
        foreach (DataGridViewRow row in _grid.Rows)
        {
            if (row.DataBoundItem is ModelVersionGridRow item && item.Id == id.Value)
            {
                row.Selected = true;
                _grid.CurrentCell = row.Cells[0];
                break;
            }
        }
    }

    private void ToggleBusy(bool busy)
    {
        UseWaitCursor = busy;
        _btnRefresh.Enabled = !busy;
        _btnClose.Enabled = !busy;
        if (busy)
        {
            _btnDetails.Enabled = false;
            _btnPublish.Enabled = false;
            _btnUnpublish.Enabled = false;
            _btnPin.Enabled = false;
            _btnUnpin.Enabled = false;
            _btnDelete.Enabled = false;
        }
        else
        {
            RenderSelected();
        }
    }

    private static string ToLocalString(DateTime value)
        => value.ToLocalTime().ToString("dd.MM.yyyy HH:mm:ss", CultureInfo.InvariantCulture);

    private static string BuildMetricsShort(string? metricsJson)
    {
        if (string.IsNullOrWhiteSpace(metricsJson)) return "-";
        try
        {
            using var document = JsonDocument.Parse(metricsJson);
            var root = document.RootElement;
            var parts = new List<string>();
            AddMetric(parts, root, "mAP50_95", "mAP");
            AddMetric(parts, root, "mAP50", "mAP50");
            AddMetric(parts, root, "precision", "P");
            AddMetric(parts, root, "recall", "R");
            return parts.Count == 0 ? "json" : string.Join(" · ", parts);
        }
        catch
        {
            return metricsJson.Length > 120 ? metricsJson[..120] + "..." : metricsJson;
        }
    }

    private static void AddMetric(List<string> parts, JsonElement root, string key, string title)
    {
        if (root.TryGetProperty(key, out var value) && value.TryGetDouble(out var number))
            parts.Add($"{title}: {number:F4}");
    }

    private static string PrettyJson(string? json)
    {
        if (string.IsNullOrWhiteSpace(json)) return "Метрики не сохранены.";
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

    private sealed class ModelVersionGridRow
    {
        public int Id { get; set; }
        public string ExternalJobId { get; set; } = string.Empty;
        public string TrainedAt { get; set; } = string.Empty;
        public string BaseModel { get; set; } = string.Empty;
        public string MobileFormat { get; set; } = string.Empty;
        public string MobileModelFileName { get; set; } = string.Empty;
        public bool IsPublished { get; set; }
        public bool IsPinned { get; set; }
        public string MetricsSummary { get; set; } = string.Empty;
        public required ModelVersionAdminResponse Source { get; set; }
    }
}
