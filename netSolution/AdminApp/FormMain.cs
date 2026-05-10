namespace AdminApp;

public partial class FormMain : Form
{
    private readonly string _token;
    private readonly AdminApiClient _api;

    public FormMain(string token)
    {
        InitializeComponent();

        _token = token;

        var settings = AdminSettings.Load();
        _api = new AdminApiClient(settings.ApiBaseUrl, _token);

        Text = "Админка";
        Width = 560;
        Height = 390;
        StartPosition = FormStartPosition.CenterScreen;

        BuildUi();
    }

    private void BuildUi()
    {
        Controls.Clear();

        var lblTitle = new Label
        {
            Text = "Панель администратора",
            AutoSize = true,
            Font = new Font("Segoe UI", 16, FontStyle.Bold),
            Location = new Point(20, 20)
        };

        var btnErrorReports = new Button
        {
            Text = "Сообщения об ошибках",
            Width = 220,
            Height = 50,
            Location = new Point(20, 80)
        };

        btnErrorReports.Click += (_, _) =>
        {
            var form = new FormErrorReports(_api);
            form.ShowDialog(this);
        };

        var btnTraining = new Button
        {
            Text = "Обучение модели",
            Width = 220,
            Height = 50,
            Location = new Point(260, 80)
        };

        btnTraining.Click += (_, _) =>
        {
            using var form = new FormTraining(_api);
            form.ShowDialog(this);
        };

        var btnModelVersions = new Button
        {
            Text = "Версии модели",
            Width = 220,
            Height = 50,
            Location = new Point(20, 150)
        };

        btnModelVersions.Click += (_, _) =>
        {
            using var form = new FormModelVersions(_api);
            form.ShowDialog(this);
        };

        var btnBackup = new Button
        {
            Text = "Backup / восстановление",
            Width = 220,
            Height = 50,
            Location = new Point(260, 150)
        };

        btnBackup.Click += (_, _) =>
        {
            using var form = new FormBackup(_api);
            form.ShowDialog(this);
        };

        Controls.Add(lblTitle);
        Controls.Add(btnErrorReports);
        Controls.Add(btnTraining);
        Controls.Add(btnModelVersions);
        Controls.Add(btnBackup);
    }
}
