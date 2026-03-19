using Contracts.Dtos;

namespace AdminApp;

public partial class FormErrorReports : Form
{
    private readonly AdminApiClient _api;
    private readonly BindingSource _binding = new();

    private DataGridView _grid = null!;
    private Button _btnRefresh = null!;
    private Button _btnOpen = null!;

    public FormErrorReports(AdminApiClient api)
    {
        InitializeComponent();

        _api = api;

        Text = "Сообщения об ошибках";
        Width = 900;
        Height = 600;
        StartPosition = FormStartPosition.CenterParent;

        BuildUi();

        Shown += async (_, _) => await LoadReportsAsync();
    }

    private void BuildUi()
    {
        Controls.Clear();

        _grid = new DataGridView
        {
            Dock = DockStyle.Top,
            Height = 480,
            ReadOnly = true,
            SelectionMode = DataGridViewSelectionMode.FullRowSelect,
            MultiSelect = false,
            AutoGenerateColumns = false,
            AllowUserToAddRows = false,
            AllowUserToDeleteRows = false
        };

        _grid.Columns.Add(new DataGridViewTextBoxColumn
        {
            HeaderText = "Id",
            DataPropertyName = "Id",
            Width = 80
        });

        _grid.Columns.Add(new DataGridViewTextBoxColumn
        {
            HeaderText = "UserId",
            DataPropertyName = "UserId",
            Width = 80
        });

        _grid.Columns.Add(new DataGridViewTextBoxColumn
        {
            HeaderText = "Дата",
            DataPropertyName = "CreatedAt",
            Width = 200
        });

        _grid.Columns.Add(new DataGridViewTextBoxColumn
        {
            HeaderText = "Кол-во кадров",
            DataPropertyName = "FramesCount",
            Width = 120
        });

        _grid.Columns.Add(new DataGridViewCheckBoxColumn
        {
            HeaderText = "Approved",
            DataPropertyName = "Approved",
            Width = 90
        });

        _grid.DataSource = _binding;
        _grid.DoubleClick += (_, _) => OpenSelected();

        _btnRefresh = new Button
        {
            Text = "Обновить",
            Width = 120,
            Height = 35,
            Location = new Point(20, 500)
        };
        _btnRefresh.Click += async (_, _) => await LoadReportsAsync();

        _btnOpen = new Button
        {
            Text = "Открыть подробнее",
            Width = 180,
            Height = 35,
            Location = new Point(160, 500)
        };
        _btnOpen.Click += (_, _) => OpenSelected();

        Controls.Add(_grid);
        Controls.Add(_btnRefresh);
        Controls.Add(_btnOpen);
    }

    private async Task LoadReportsAsync()
    {
        try
        {
            UseWaitCursor = true;
            var items = await _api.GetErrorReportsAsync();
            _binding.DataSource = items;
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
        if (_binding.Current is not AdminErrorReportListItemDto item)
            return;

        var form = new FormErrorReportDetails(_api, item.Id);
        form.ShowDialog(this);
    }
}