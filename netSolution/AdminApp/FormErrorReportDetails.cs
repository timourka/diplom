using Contracts.Dtos;

namespace AdminApp;

public partial class FormErrorReportDetails : Form
{
    private readonly AdminApiClient _api;
    private readonly int _reportId;

    private AdminErrorReportDetailsDto? _report;
    private int _currentFrameIndex = 1;

    private Label _lblInfo = null!;
    private Label _lblFrameNumber = null!;
    private TextBox _txtComment = null!;
    private PictureBox _pictureBox = null!;
    private Button _btnPrev = null!;
    private Button _btnNext = null!;
    private Button _btnApprove = null!;
    private Button _btnDelete = null!;
    private Button _btnBlockUser = null!;
    private Button _btnRefresh = null!;

    private Image? _currentImage;
    private List<YoloBboxDto> _currentBboxes = new();

    public FormErrorReportDetails(AdminApiClient api, int reportId)
    {
        InitializeComponent();

        _api = api;
        _reportId = reportId;

        Text = $"Сообщение #{reportId}";
        Width = 1100;
        Height = 850;
        StartPosition = FormStartPosition.CenterParent;

        BuildUi();

        Shown += async (_, _) => await LoadAllAsync();
    }

    private void BuildUi()
    {
        Controls.Clear();

        _lblInfo = new Label
        {
            AutoSize = false,
            Width = 1040,
            Height = 80,
            Location = new Point(20, 20),
            Font = new Font("Segoe UI", 10, FontStyle.Regular)
        };

        _txtComment = new TextBox
        {
            Location = new Point(20, 110),
            Width = 1040,
            Height = 70,
            Multiline = true,
            ReadOnly = true
        };

        _pictureBox = new PictureBox
        {
            Location = new Point(20, 200),
            Width = 800,
            Height = 600,
            BorderStyle = BorderStyle.FixedSingle,
            SizeMode = PictureBoxSizeMode.Zoom
        };
        _pictureBox.Paint += PictureBox_Paint;

        _lblFrameNumber = new Label
        {
            Location = new Point(840, 220),
            Width = 220,
            Height = 30,
            Font = new Font("Segoe UI", 10, FontStyle.Bold)
        };

        _btnPrev = new Button
        {
            Text = "←",
            Width = 80,
            Height = 40,
            Location = new Point(840, 270)
        };
        _btnPrev.Click += async (_, _) =>
        {
            if (_report is null || _currentFrameIndex <= 1) return;
            _currentFrameIndex--;
            await LoadFrameAsync();
        };

        _btnNext = new Button
        {
            Text = "→",
            Width = 80,
            Height = 40,
            Location = new Point(940, 270)
        };
        _btnNext.Click += async (_, _) =>
        {
            if (_report is null || _currentFrameIndex >= _report.FramesCount) return;
            _currentFrameIndex++;
            await LoadFrameAsync();
        };

        _btnApprove = new Button
        {
            Text = "Approve / Unapprove",
            Width = 220,
            Height = 40,
            Location = new Point(840, 350)
        };
        _btnApprove.Click += async (_, _) => await ToggleApproveAsync();

        _btnDelete = new Button
        {
            Text = "Удалить репорт",
            Width = 220,
            Height = 40,
            Location = new Point(840, 410)
        };
        _btnDelete.Click += async (_, _) => await DeleteReportAsync();

        _btnBlockUser = new Button
        {
            Text = "Заблокировать пользователя",
            Width = 220,
            Height = 40,
            Location = new Point(840, 470)
        };
        _btnBlockUser.Click += async (_, _) => await BlockUserAsync();

        _btnRefresh = new Button
        {
            Text = "Обновить",
            Width = 220,
            Height = 40,
            Location = new Point(840, 530)
        };
        _btnRefresh.Click += async (_, _) => await LoadAllAsync();

        Controls.Add(_lblInfo);
        Controls.Add(_txtComment);
        Controls.Add(_pictureBox);
        Controls.Add(_lblFrameNumber);
        Controls.Add(_btnPrev);
        Controls.Add(_btnNext);
        Controls.Add(_btnApprove);
        Controls.Add(_btnDelete);
        Controls.Add(_btnBlockUser);
        Controls.Add(_btnRefresh);
    }

    private async Task LoadAllAsync()
    {
        try
        {
            UseWaitCursor = true;

            _report = await _api.GetErrorReportAsync(_reportId);
            if (_report == null)
            {
                MessageBox.Show(this, "Репорт не найден", "Ошибка", MessageBoxButtons.OK, MessageBoxIcon.Error);
                Close();
                return;
            }

            if (_report.FramesCount <= 0)
            {
                _currentFrameIndex = 0;
            }
            else
            {
                if (_currentFrameIndex < 1) _currentFrameIndex = 1;
                if (_currentFrameIndex > _report.FramesCount) _currentFrameIndex = _report.FramesCount;
            }

            _lblInfo.Text =
                $"ReportId: {_report.Id}\r\n" +
                $"UserId: {_report.UserId}\r\n" +
                $"CreatedAt: {_report.CreatedAt}\r\n" +
                $"FramesCount: {_report.FramesCount}\r\n" +
                $"Approved: {_report.Approved}";

            _txtComment.Text = _report.Comment ?? "";

            await LoadFrameAsync();
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

    private async Task LoadFrameAsync()
    {
        if (_report == null) return;

        if (_report.FramesCount <= 0)
        {
            _currentImage?.Dispose();
            _currentImage = null;
            _currentBboxes = new List<YoloBboxDto>();
            _pictureBox.Image = null;
            _lblFrameNumber.Text = "Нет кадров для просмотра";
            UpdateNavigationButtons();
            _pictureBox.Invalidate();
            return;
        }

        try
        {
            UseWaitCursor = true;

            _currentImage?.Dispose();
            _currentImage = await _api.GetFrameImageAsync(_report.Id, _currentFrameIndex);
            _currentBboxes = await _api.GetFrameBboxesAsync(_report.Id, _currentFrameIndex);

            _pictureBox.Image = _currentImage;
            _lblFrameNumber.Text = $"Кадр {_currentFrameIndex} / {_report.FramesCount}";
            UpdateNavigationButtons();
            _pictureBox.Invalidate();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка загрузки кадра", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            UseWaitCursor = false;
        }
    }

    private void UpdateNavigationButtons()
    {
        var hasFrames = _report is not null && _report.FramesCount > 0;
        _btnPrev.Enabled = hasFrames && _currentFrameIndex > 1;
        _btnNext.Enabled = hasFrames && _report is not null && _currentFrameIndex < _report.FramesCount;
    }

    private async Task ToggleApproveAsync()
    {
        if (_report == null) return;

        try
        {
            await _api.SetReportApprovedAsync(_report.Id, !_report.Approved);
            await LoadAllAsync();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    private async Task DeleteReportAsync()
    {
        if (_report == null) return;

        var confirm = MessageBox.Show(
            this,
            $"Удалить репорт #{_report.Id}?",
            "Подтверждение",
            MessageBoxButtons.YesNo,
            MessageBoxIcon.Warning);

        if (confirm != DialogResult.Yes)
            return;

        try
        {
            await _api.DeleteReportAsync(_report.Id);
            MessageBox.Show(this, "Репорт удалён", "OK", MessageBoxButtons.OK, MessageBoxIcon.Information);
            Close();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка удаления", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    private async Task BlockUserAsync()
    {
        if (_report == null) return;

        var confirm = MessageBox.Show(
            this,
            $"Заблокировать пользователя #{_report.UserId}?",
            "Подтверждение",
            MessageBoxButtons.YesNo,
            MessageBoxIcon.Warning);

        if (confirm != DialogResult.Yes)
            return;

        try
        {
            await _api.BlockUserAsync(_report.UserId, true);
            MessageBox.Show(this, "Пользователь заблокирован", "OK", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка блокировки", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    private void PictureBox_Paint(object? sender, PaintEventArgs e)
    {
        if (_pictureBox.Image == null)
        {
            using var brush = new SolidBrush(Color.DimGray);
            using var font = new Font("Segoe UI", 12, FontStyle.Regular);
            var text = _report is null || _report.FramesCount <= 0
                ? "Нет кадров для просмотра"
                : "Кадр отсутствует или был исключён";
            var format = new StringFormat
            {
                Alignment = StringAlignment.Center,
                LineAlignment = StringAlignment.Center
            };
            e.Graphics.DrawString(text, font, brush, _pictureBox.ClientRectangle, format);
            return;
        }

        if (_currentBboxes.Count == 0)
            return;

        var img = _pictureBox.Image;
        var imgW = img.Width;
        var imgH = img.Height;
        var rect = GetImageDisplayRectangle(_pictureBox);

        float scaleX = rect.Width / (float)imgW;
        float scaleY = rect.Height / (float)imgH;

        using var pen = new Pen(Color.Red, 3);

        foreach (var bbox in _currentBboxes)
        {
            var bboxW = (float)(bbox.W * imgW);
            var bboxH = (float)(bbox.H * imgH);
            var centerX = (float)(bbox.Xc * imgW);
            var centerY = (float)(bbox.Yc * imgH);

            var left = centerX - bboxW / 2f;
            var top = centerY - bboxH / 2f;

            var drawLeft = rect.X + left * scaleX;
            var drawTop = rect.Y + top * scaleY;
            var drawWidth = bboxW * scaleX;
            var drawHeight = bboxH * scaleY;

            e.Graphics.DrawRectangle(pen, drawLeft, drawTop, drawWidth, drawHeight);
        }
    }

    private static Rectangle GetImageDisplayRectangle(PictureBox pb)
    {
        if (pb.Image == null)
            return pb.ClientRectangle;

        var img = pb.Image;
        float imageRatio = (float)img.Width / img.Height;
        float boxRatio = (float)pb.ClientSize.Width / pb.ClientSize.Height;

        if (imageRatio > boxRatio)
        {
            int width = pb.ClientSize.Width;
            int height = (int)(width / imageRatio);
            int top = (pb.ClientSize.Height - height) / 2;
            return new Rectangle(0, top, width, height);
        }
        else
        {
            int height = pb.ClientSize.Height;
            int width = (int)(height * imageRatio);
            int left = (pb.ClientSize.Width - width) / 2;
            return new Rectangle(left, 0, width, height);
        }
    }
}