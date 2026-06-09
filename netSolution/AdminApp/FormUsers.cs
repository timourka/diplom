using Contracts.Dtos;

namespace AdminApp;

public sealed class FormUsers : Form
{
    private readonly AdminApiClient _api;
    private readonly BindingSource _binding = new();

    private DataGridView _grid = null!;
    private Button _btnRefresh = null!;
    private Button _btnOpen = null!;
    private Button _btnBlockToggle = null!;
    private Button _btnDelete = null!;
    private Button _btnClose = null!;

    public FormUsers(AdminApiClient api)
    {
        _api = api;

        Text = "Управление пользователями";
        Width = 980;
        Height = 640;
        StartPosition = FormStartPosition.CenterParent;

        BuildUi();

        Shown += async (_, _) => await LoadUsersAsync();
    }

    private void BuildUi()
    {
        Controls.Clear();

        _grid = new DataGridView
        {
            Dock = DockStyle.Top,
            Height = 500,
            ReadOnly = true,
            SelectionMode = DataGridViewSelectionMode.FullRowSelect,
            MultiSelect = false,
            AutoGenerateColumns = false,
            AllowUserToAddRows = false,
            AllowUserToDeleteRows = false,
            RowHeadersVisible = false
        };

        _grid.Columns.Add(new DataGridViewTextBoxColumn
        {
            HeaderText = "Id",
            DataPropertyName = nameof(AdminUserListItem.Id),
            Width = 80
        });

        _grid.Columns.Add(new DataGridViewTextBoxColumn
        {
            HeaderText = "Логин",
            DataPropertyName = nameof(AdminUserListItem.Login),
            AutoSizeMode = DataGridViewAutoSizeColumnMode.Fill
        });

        _grid.Columns.Add(new DataGridViewCheckBoxColumn
        {
            HeaderText = "Заблокирован",
            DataPropertyName = nameof(AdminUserListItem.IsBlocked),
            Width = 120
        });

        _grid.Columns.Add(new DataGridViewCheckBoxColumn
        {
            HeaderText = "Админ",
            DataPropertyName = nameof(AdminUserListItem.IsAdmin),
            Width = 90
        });

        _grid.Columns.Add(new DataGridViewTextBoxColumn
        {
            HeaderText = "Создан",
            DataPropertyName = nameof(AdminUserListItem.CreatedAt),
            Width = 170
        });

        _grid.DataSource = _binding;
        _grid.DoubleClick += (_, _) => OpenSelected();
        _grid.SelectionChanged += (_, _) => UpdateActionButtons();

        var actions = new FlowLayoutPanel
        {
            Dock = DockStyle.Bottom,
            Height = 70,
            Padding = new Padding(20, 14, 20, 12),
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false
        };

        _btnRefresh = new Button { Text = "Обновить", Width = 120, Height = 35 };
        _btnRefresh.Click += async (_, _) => await LoadUsersAsync();

        _btnOpen = new Button { Text = "Открыть подробнее", Width = 180, Height = 35 };
        _btnOpen.Click += (_, _) => OpenSelected();

        _btnBlockToggle = new Button { Text = "Блокировка", Width = 160, Height = 35 };
        _btnBlockToggle.Click += async (_, _) => await ToggleBlockedAsync();

        _btnDelete = new Button { Text = "Удалить", Width = 120, Height = 35 };
        _btnDelete.Click += async (_, _) => await DeleteSelectedAsync();

        _btnClose = new Button { Text = "Закрыть", Width = 120, Height = 35 };
        _btnClose.Click += (_, _) => Close();

        actions.Controls.AddRange([_btnRefresh, _btnOpen, _btnBlockToggle, _btnDelete, _btnClose]);

        Controls.Add(_grid);
        Controls.Add(actions);
    }

    private async Task LoadUsersAsync()
    {
        try
        {
            UseWaitCursor = true;
            var users = await _api.GetUsersAsync();
            _binding.DataSource = users;
            UpdateActionButtons();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            UseWaitCursor = false;
        }
    }

    private void OpenSelected()
    {
        if (_binding.Current is not AdminUserListItem item)
            return;

        using var form = new FormUserDetails(_api, item.Id);
        form.ShowDialog(this);
        _ = LoadUsersAsync();
    }

    private async Task ToggleBlockedAsync()
    {
        if (_binding.Current is not AdminUserListItem item)
            return;

        var newBlockedState = !item.IsBlocked;
        var actionText = newBlockedState ? "заблокировать" : "разблокировать";

        var confirm = MessageBox.Show(
            this,
            $"{Capitalize(actionText)} пользователя {item.Login}?",
            "Подтверждение",
            MessageBoxButtons.YesNo,
            MessageBoxIcon.Question);

        if (confirm != DialogResult.Yes)
            return;

        try
        {
            UseWaitCursor = true;
            await _api.BlockUserAsync(item.Id, newBlockedState);
            await LoadUsersAsync();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            UseWaitCursor = false;
        }
    }

    private async Task DeleteSelectedAsync()
    {
        if (_binding.Current is not AdminUserListItem item)
            return;

        var confirm = MessageBox.Show(
            this,
            $"Удалить пользователя {item.Login}?\r\n\r\nЕго связанные данные также могут быть удалены согласно правилам БД.",
            "Подтверждение удаления",
            MessageBoxButtons.YesNo,
            MessageBoxIcon.Warning);

        if (confirm != DialogResult.Yes)
            return;

        try
        {
            UseWaitCursor = true;
            await _api.DeleteUserAsync(item.Id);
            await LoadUsersAsync();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            UseWaitCursor = false;
        }
    }

    private void UpdateActionButtons()
    {
        var hasSelection = _binding.Current is AdminUserListItem;
        _btnOpen.Enabled = hasSelection;
        _btnBlockToggle.Enabled = hasSelection;
        _btnDelete.Enabled = hasSelection;

        if (_binding.Current is AdminUserListItem item)
            _btnBlockToggle.Text = item.IsBlocked ? "Разблокировать" : "Заблокировать";
        else
            _btnBlockToggle.Text = "Блокировка";
    }

    private static string Capitalize(string value)
        => string.IsNullOrEmpty(value) ? value : char.ToUpper(value[0]) + value[1..];
}
