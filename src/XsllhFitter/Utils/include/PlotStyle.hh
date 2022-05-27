#ifndef PLOTSTYLE_HH
#define PLOTSTYLE_HH

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TLegend.h"
#include "TMatrixT.h"
#include "TMatrixTSym.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TTree.h"

#include "nlohmann/json.hpp"
using json = nlohmann::json;

#include "ColorOutput.hh"

class GlobalStyle
{
    public:
        GlobalStyle();
        GlobalStyle(const std::string& json_config);

        void ApplyStyle();
        void SetDefault();

        int frame_border_mode;
        int canvas_border_mode;
        int pad_border_mode;
        int pad_color;
        int canvas_color;
        int stat_color;
        int line_width;

        int legend_border_size;
        int legend_fill_color;

        float pad_top_margin;
        float pad_right_margin;
        float pad_bottom_margin;
        float pad_left_margin;

        int global_font;
        int text_font;
        float text_size;
        int label_font;
        float label_size;
        int title_font;
        float title_size;
        int legend_font;
        float legend_text_size;

        bool do_pad_ticks;
        bool do_strip_decimals;
        bool do_grid_lines;
        int n_divisions;

    private:
        TStyle custom_style;
};

class HistStyle
{
    public:
        HistStyle();
        HistStyle(const std::string& json_config);
        HistStyle(json& j);

        void ApplyStyle(TH1D& hist);
        void ApplyStyle(std::vector<TH1D>& v_hist);

        void SetDefault();
        void SetAxisTitle(const std::string& x_str, const std::string& y_str);
        void SetAxisRange(double x1, double x2, double y1, double y2);
        void ScaleXbins(TH1D& hist, double scale_factor);
        void SetFillAtt(int f_color, int f_style);
        void SetLineAtt(int l_color, int l_style, int l_width);
        void SetMarkerAtt(int m_color, int m_style, float m_size);

    private:
        std::string x_axis_title;
        std::string y_axis_title;
        int fill_color;
        int fill_style;
        int line_color;
        int line_style;
        int line_width;
        int marker_color;
        int marker_style;
        float marker_size;
        double x_lo, x_up;
        double y_lo, y_up;
};

#endif
