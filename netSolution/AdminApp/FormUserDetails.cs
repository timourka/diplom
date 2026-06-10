using Contracts.Dtos;

namespace AdminApp;

public sealed class FormUserDetails : Form
{
    private readonly AdminApiClient _api;
    private readonly int _userId;

    private AdminUserDetailsDto? _user;

    private Label _lblIdValue = null!;
    private Label _lblCreatedAtValue = null!;
    private Label _lblStoredProductsCountValue = null!;
    private Label _lblErrorReportsCountValue = null!;
    private Label _lblApprovedReportsCountValue = null!;
    private TextBox _txtLogin = null!;
    private CheckBox _chkBlocked = null!;
    private CheckBox _chkAdmin = null!;
    private Button _btnSave = null!;
    private Button _btnRefresh = null!;
    private Button _btnDelete = null!;
    private Button _btnClose = null!;

    public FormUserDetails(AdminApiClient api, int userId)
    {
        _api = api;
        _userId = userId;

        Text = $"Пользователь #{userId}";
        Width = 760;
        Height = 620;
        StartPosition = FormStartPosition.CenterParent;

        BuildUi();

        Shown += async (_, _) => await LoadUserAsync();
    }

    private void BuildUi()
    {
        Controls.Clear();

        var root = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            Padding = new Padding(16),
            ColumnCount = 2,
            RowCount = 12
        };

        root.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 190));
        root.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

        _lblIdValue = CreateValueLabel();
        _lblCreatedAtValue = CreateValueLabel();
        _lblStoredProductsCountValue = CreateValueLabel();
        _lblErrorReportsCountValue = CreateValueLabel();
        _lblApprovedReportsCountValue = CreateValueLabel();

        _txtLogin = new TextBox { Dock = DockStyle.Fill };
        _chkBlocked = new CheckBox { Text = "Пользователь заблокирован", AutoSize = true };
        _chkAdmin = new CheckBox { Text = "Административный профиль", AutoSize = true };

        AddRow(root, 0, "Id", _lblIdValue);
        AddRow(root, 1, "Логин", _txtLogin);
        AddRow(root, 2, "Создан", _lblCreatedAtValue);
        AddRow(root, 3, "Блокировка", _chkBlocked);
        AddRow(root, 4, "Права", _chkAdmin);
        AddRow(root, 5, "Товаров на хранении", _lblStoredProductsCountValue);
        AddRow(root, 6, "Отчётов об ошибках", _lblErrorReportsCountValue);
        AddRow(root, 7, "Одобренных отчётов", _lblApprovedReportsCountValue);

        root.RowStyles.Clear();
        for (var i = 0; i < 8; i++)
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 38));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 0));
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 0));
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 0));

        var actions = new FlowLayoutPanel
        {
            Dock = DockStyle.Bottom,
            Height = 66,
            Padding = new Padding(16, 10, 16, 12),
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false
        };

        _btnSave = new Button { Text = "Сохранить", Width = 130, Height = 36 };
        _btnSave.Click += async (_, _) => await SaveAsync();

        _btnRefresh = new Button { Text = "Обновить", Width = 120, Height = 36 };
        _btnRefresh.Click += async (_, _) => await LoadUserAsync();

        _btnDelete = new Button { Text = "Удалить", Width = 120, Height = 36 };
        _btnDelete.Click += async (_, _) => await DeleteAsync();

        _btnClose = new Button { Text = "Закрыть", Width = 120, Height = 36 };
        _btnClose.Click += (_, _) => Close();

        actions.Controls.AddRange([_btnSave, _btnRefresh, _btnDelete, _btnClose]);

        Controls.Add(root);
        Controls.Add(actions);
    }

    private async Task LoadUserAsync()
    {
        try
        {
            ToggleBusy(true);
            _user = await _api.GetUserAsync(_userId);

            if (_user is null)
            {
                MessageBox.Show(this, "Пользователь не найден.", "Пользователь", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                DialogResult = DialogResult.OK;
                Close();
                return;
            }

            Render(_user);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            ToggleBusy(false);
        }
    }

    private void Render(AdminUserDetailsDto user)
    {
        _lblIdValue.Text = user.Id.ToString();
        _txtLogin.Text = user.Login;
        _lblCreatedAtValue.Text = FormatDate(user.CreatedAt);
        _chkBlocked.Checked = user.IsBlocked;
        _chkAdmin.Checked = user.IsAdmin;
        _lblStoredProductsCountValue.Text = user.StoredProductsCount.ToString();
        _lblErrorReportsCountValue.Text = user.ErrorReportsCount.ToString();
        _lblApprovedReportsCountValue.Text = user.ApprovedReportsCount.ToString();
    }

    private async Task SaveAsync()
    {
        if (_user is null)
            return;

        var login = _txtLogin.Text.Trim();
        if (string.IsNullOrWhiteSpace(login))
        {
            MessageBox.Show(this, "Логин не может быть пустым.", "Проверка", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            _txtLogin.Focus();
            return;
        }

        try
        {
            ToggleBusy(true);

            _user = await _api.UpdateUserAsync(
                _user.Id,
                new AdminUserUpdateRequest(
                    login,
                    _chkBlocked.Checked,
                    _chkAdmin.Checked,
                    null
                )
            );

            Render(_user);
            MessageBox.Show(this, "Пользователь сохранён.", "OK", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка сохранения", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            ToggleBusy(false);
        }
    }

    private async Task DeleteAsync()
    {
        if (_user is null)
            return;

        var confirm = MessageBox.Show(
            this,
            $"Удалить пользователя {_user.Login}?\r\n\r\nЕго связанные данные также могут быть удалены согласно правилам БД.",
            "Подтверждение удаления",
            MessageBoxButtons.YesNo,
            MessageBoxIcon.Warning);

        if (confirm != DialogResult.Yes)
            return;

        try
        {
            ToggleBusy(true);
            await _api.DeleteUserAsync(_user.Id);
            DialogResult = DialogResult.OK;
            Close();
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

    private void ToggleBusy(bool busy)
    {
        UseWaitCursor = busy;
        _btnSave.Enabled = !busy;
        _btnRefresh.Enabled = !busy;
        _btnDelete.Enabled = !busy;
        _btnClose.Enabled = !busy;
        _txtLogin.Enabled = !busy;
        _chkBlocked.Enabled = !busy;
        _chkAdmin.Enabled = !busy;
    }

    private static void AddRow(TableLayoutPanel panel, int rowIndex, string caption, Control control, int rowSpan = 1)
    {
        var label = new Label
        {
            Text = caption,
            Dock = DockStyle.Fill,
            TextAlign = ContentAlignment.MiddleLeft,
            Font = new Font("Segoe UI", 9, FontStyle.Bold)
        };

        control.Margin = new Padding(3, 4, 3, 4);

        panel.Controls.Add(label, 0, rowIndex);
        panel.Controls.Add(control, 1, rowIndex);

        if (rowSpan > 1)
            panel.SetRowSpan(control, rowSpan);
    }

    private static Label CreateValueLabel() => new()
    {
        Dock = DockStyle.Fill,
        TextAlign = ContentAlignment.MiddleLeft,
        AutoEllipsis = true
    };

    private static string FormatDate(DateTime? value)
        => value.HasValue ? value.Value.ToLocalTime().ToString("dd.MM.yyyy HH:mm") : "-";
}
