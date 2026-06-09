namespace AdminApp;

public partial class FormLogin : Form
{
    private readonly AdminSettings _settings;

    public string? Token { get; private set; }

    private TextBox _txtEmail = null!;
    private TextBox _txtPassword = null!;
    private Button _btnLogin = null!;

    public FormLogin()
    {
        _settings = AdminSettings.Load();

        InitializeComponent();

        Text = "Вход в админку";
        Width = 400;
        Height = 220;
        StartPosition = FormStartPosition.CenterScreen;

        BuildUi();
    }

    private void BuildUi()
    {
        Controls.Clear();

        var lblEmail = new Label
        {
            Text = "Логин",
            Left = 20,
            Top = 20,
            Width = 100
        };

        _txtEmail = new TextBox
        {
            Left = 20,
            Top = 45,
            Width = 320
        };

        var lblPassword = new Label
        {
            Text = "Пароль",
            Left = 20,
            Top = 80,
            Width = 100
        };

        _txtPassword = new TextBox
        {
            Left = 20,
            Top = 105,
            Width = 320,
            UseSystemPasswordChar = true
        };

        _btnLogin = new Button
        {
            Text = "Войти",
            Left = 20,
            Top = 145,
            Width = 120,
            Height = 35
        };

        _btnLogin.Click += async (_, _) => await LoginAsync();

        Controls.Add(lblEmail);
        Controls.Add(_txtEmail);
        Controls.Add(lblPassword);
        Controls.Add(_txtPassword);
        Controls.Add(_btnLogin);
    }

    private async Task LoginAsync()
    {
        try
        {
            UseWaitCursor = true;
            _btnLogin.Enabled = false;

            var token = await AdminApiClient.LoginAsync(
                _settings.ApiBaseUrl,
                _txtEmail.Text.Trim(),
                _txtPassword.Text
            );

            Token = token;
            DialogResult = DialogResult.OK;
            Close();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка входа", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            UseWaitCursor = false;
            _btnLogin.Enabled = true;
        }
    }
}