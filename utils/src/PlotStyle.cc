#include "PlotStyle.hh"

GlobalStyle::GlobalStyle()
    : custom_style("xsllh_style", "Super xsllh Style")
{
    SetDefault();
}

GlobalStyle::GlobalStyle(const std::string& json_config)
    : custom_style("xsllh_style", "Super xsllh Style")
{
}

void GlobalStyle::SetDefault()
{
    frame_border_mode = 0;
    canvas_border_mode = 0;
    pad_border_mode = 0;
    pad_color = 0;
    canvas_color = 0;
    stat_color = 0;
    line_width = 2;

    legend_border_size = 0;
    legend_fill_color = 0;

    pad_top_margin = 0.14;
    pad_right_margin = 0.14;
    pad_bottom_margin = 0.14;
    pad_left_margin = 0.14;

    global_font = 42;
    text_font = global_font;
    text_size = 0.08;
    label_font = global_font;
    label_size = 0.05;
    title_font = global_font;
    title_size = 0.06;
    legend_font = global_font;
    legend_text_size = 0.04;

    do_pad_ticks = true;
    do_strip_decimals = false;
    do_grid_lines = false;
    n_divisions = 510;
}

void GlobalStyle::ApplyStyle()
{
    custom_style.SetFrameBorderMode(frame_border_mode);
    custom_style.SetCanvasBorderMode(canvas_border_mode);
    custom_style.SetPadBorderMode(pad_border_mode);
    custom_style.SetPadColor(pad_color);
    custom_style.SetCanvasColor(canvas_color);
    custom_style.SetStatColor(stat_color);
    custom_style.SetLineWidth(line_width);

    custom_style.SetLegendBorderSize(legend_border_size);
    custom_style.SetLegendFillColor(legend_fill_color);

    custom_style.SetPadTopMargin(pad_top_margin);
    custom_style.SetPadRightMargin(pad_right_margin);
    custom_style.SetPadBottomMargin(pad_bottom_margin);
    custom_style.SetPadLeftMargin(pad_left_margin);

    custom_style.SetTextFont(text_font);
    custom_style.SetTextSize(text_size);
    custom_style.SetLabelFont(label_font,"xyzt");
    custom_style.SetLabelSize(label_size,"xyzt");
    custom_style.SetTitleFont(title_font,"xyzt");
    custom_style.SetTitleSize(title_size,"xyzt");
    custom_style.SetLegendFont(legend_font);
    custom_style.SetLegendTextSize(legend_text_size);

    custom_style.SetStripDecimals(do_strip_decimals);
    custom_style.SetPadTickX(do_pad_ticks);
    custom_style.SetPadTickY(do_pad_ticks);
    custom_style.SetNdivisions(n_divisions, "xy");

    custom_style.SetOptTitle(0);
    custom_style.SetOptStat(0);
    custom_style.SetOptFit(0);

    gROOT -> SetStyle("xsllh_style");
    gROOT -> ForceStyle();
    custom_style.cd();
}

HistStyle::HistStyle()
{
    SetDefault();
}

HistStyle::HistStyle(const std::string& json_config)
{
}

HistStyle::HistStyle(json& j)
{
}

void HistStyle::SetDefault()
{
    x_axis_title = "";
    y_axis_title = "";
    fill_color = kBlack;
    fill_style = 0;
    line_color = kBlack;
    line_style = kSolid;
    line_width = 2;
    marker_color = kBlack;
    marker_style = kFullCircle;
    marker_size = 1;

    x_lo = 0;
    x_up = 0;
    y_lo = 0;
    y_up = 0;
}

void HistStyle::ApplyStyle(TH1D& hist)
{
    hist.GetXaxis()->SetTitle(x_axis_title.c_str());
    hist.GetYaxis()->SetTitle(y_axis_title.c_str());
    hist.SetFillColor(fill_color);
    hist.SetFillStyle(fill_style);
    hist.SetLineColor(line_color);
    hist.SetLineStyle(line_style);
    hist.SetLineWidth(line_width);
    hist.SetMarkerColor(marker_color);
    hist.SetMarkerStyle(marker_style);
    hist.SetMarkerSize(marker_size);

    if(x_lo != x_up)
        hist.GetXaxis()->SetRangeUser(x_lo, x_up);

    if(y_lo != y_up)
        hist.GetYaxis()->SetRangeUser(y_lo, y_up);
}

void HistStyle::ApplyStyle(std::vector<TH1D>& v_hist)
{
    for(auto& h : v_hist)
        ApplyStyle(h);
}

void HistStyle::SetAxisTitle(const std::string& x_str, const std::string& y_str)
{
    x_axis_title = x_str;
    y_axis_title = y_str;
}

void HistStyle::SetAxisRange(double x1, double x2, double y1, double y2)
{
    x_lo = x1;
    x_up = x2;
    y_lo = y1;
    y_up = y2;
}

void HistStyle::ScaleXbins(TH1D& hist, double scale_factor)
{
    auto root_array = hist.GetXaxis()->GetXbins();
    double* axis_bins = const_cast<double*>(root_array->GetArray());
    for(int i = 0; i < hist.GetNbinsX()+1; ++i)
        axis_bins[i] = axis_bins[i] * scale_factor;
    hist.GetXaxis()->Set(hist.GetNbinsX(), axis_bins);
}

void HistStyle::SetFillAtt(int f_color, int f_style)
{
    fill_color = f_color;
    fill_style = f_style;
}

void HistStyle::SetLineAtt(int l_color, int l_style, int l_width)
{
    line_color = l_color;
    line_style = l_style;
    line_width = l_width;
}

void HistStyle::SetMarkerAtt(int m_color, int m_style, float m_size)
{
    marker_color = m_color;
    marker_style = m_style;
    marker_size = m_size;
}
