namespace AdminApp;

public sealed class FormBackup : Form
{
    private readonly AdminApiClient _api;
    private readonly Label _lblStatus = new();
    private readonly Button _btnExport = new();
    private readonly Button _btnImport = new();
    private readonly CheckBox _chkReplaceExisting = new();

    public FormBackup(AdminApiClient api)
    {
        _api = api;

        Text = "Резервная копия";
        Width = 620;
        Height = 330;
        StartPosition = FormStartPosition.CenterParent;
        FormBorderStyle = FormBorderStyle.FixedDialog;
        MaximizeBox = false;
        MinimizeBox = false;

        BuildUi();
    }

    private void BuildUi()
    {
        Controls.Clear();

        var title = new Label
        {
            Text = "Экспорт / импорт данных",
            AutoSize = true,
            Font = new Font("Segoe UI", 15, FontStyle.Bold),
            Location = new Point(20, 20)
        };

        var description = new Label
        {
            Text = "Backup ZIP содержит базу данных, модели, датасеты обучения и сообщения об ошибках.",
            AutoSize = false,
            Width = 560,
            Height = 45,
            Location = new Point(20, 60)
        };

        _btnExport.Text = "Экспортировать в ZIP";
        _btnExport.Width = 230;
        _btnExport.Height = 44;
        _btnExport.Location = new Point(20, 120);
        _btnExport.Click += async (_, _) => await ExportAsync();

        _btnImport.Text = "Импортировать ZIP";
        _btnImport.Width = 230;
        _btnImport.Height = 44;
        _btnImport.Location = new Point(270, 120);
        _btnImport.Click += async (_, _) => await ImportAsync();

        _chkReplaceExisting.Text = "При импорте полностью заменить текущую базу и файлы";
        _chkReplaceExisting.AutoSize = true;
        _chkReplaceExisting.Location = new Point(20, 185);

        var warning = new Label
        {
            Text = "Внимание: режим полной замены удалит текущие записи, модели и загруженные файлы на backend перед восстановлением из ZIP.",
            AutoSize = false,
            Width = 560,
            Height = 45,
            Location = new Point(40, 212)
        };

        _lblStatus.Text = "Готово.";
        _lblStatus.AutoSize = false;
        _lblStatus.Width = 560;
        _lblStatus.Height = 36;
        _lblStatus.Location = new Point(20, 255);

        Controls.Add(title);
        Controls.Add(description);
        Controls.Add(_btnExport);
        Controls.Add(_btnImport);
        Controls.Add(_chkReplaceExisting);
        Controls.Add(warning);
        Controls.Add(_lblStatus);
    }

    private async Task ExportAsync()
    {
        using var dialog = new SaveFileDialog
        {
            Title = "Сохранить backup ZIP",
            Filter = "ZIP archive (*.zip)|*.zip",
            FileName = $"productsdate_backup_{DateTime.Now:yyyyMMdd_HHmmss}.zip",
            AddExtension = true,
            DefaultExt = "zip",
            OverwritePrompt = true
        };

        if (dialog.ShowDialog(this) != DialogResult.OK)
            return;

        await RunOperationAsync(async () =>
        {
            _lblStatus.Text = "Экспортирую backup с сервера...";
            await _api.ExportBackupAsync(dialog.FileName);
            _lblStatus.Text = $"Backup сохранён: {dialog.FileName}";
            MessageBox.Show(this, "Backup успешно экспортирован.", "Экспорт", MessageBoxButtons.OK, MessageBoxIcon.Information);
        });
    }

    private async Task ImportAsync()
    {
        using var dialog = new OpenFileDialog
        {
            Title = "Выбрать backup ZIP",
            Filter = "ZIP archive (*.zip)|*.zip",
            CheckFileExists = true,
            Multiselect = false
        };

        if (dialog.ShowDialog(this) != DialogResult.OK)
            return;

        var replaceExisting = _chkReplaceExisting.Checked;
        var message = replaceExisting
            ? "Импорт с полной заменой удалит текущую базу и файлы на сервере. Продолжить?"
            : "Импорт будет выполнен только если база на сервере пустая. Продолжить?";

        var confirm = MessageBox.Show(this, message, "Подтверждение импорта", MessageBoxButtons.YesNo, MessageBoxIcon.Warning);
        if (confirm != DialogResult.Yes)
            return;

        await RunOperationAsync(async () =>
        {
            _lblStatus.Text = "Загружаю backup на сервер и восстанавливаю данные...";
            var result = await _api.ImportBackupAsync(dialog.FileName, replaceExisting);
            _lblStatus.Text = BuildImportSummary(result);
            MessageBox.Show(this, BuildImportSummary(result), "Импорт завершён", MessageBoxButtons.OK, MessageBoxIcon.Information);
        });
    }

    private async Task RunOperationAsync(Func<Task> operation)
    {
        SetBusy(true);
        try
        {
            await operation();
        }
        catch (Exception ex)
        {
            _lblStatus.Text = "Ошибка.";
            MessageBox.Show(this, ex.Message, "Ошибка", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            SetBusy(false);
        }
    }

    private void SetBusy(bool busy)
    {
        _btnExport.Enabled = !busy;
        _btnImport.Enabled = !busy;
        _chkReplaceExisting.Enabled = !busy;
        Cursor = busy ? Cursors.WaitCursor : Cursors.Default;
    }

    private static string BuildImportSummary(BackupImportResultDto result)
    {
        return $"Импорт завершён. Пользователи: {result.Users}, продукты: {result.Products}, " +
               $"запасы: {result.StoredProducts}, видео: {result.VideoSamples}, ошибки: {result.ErrorReports}, " +
               $"версии моделей: {result.ModelVersions}, задачи обучения: {result.TrainingJobs}.";
    }
}
