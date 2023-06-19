function fig_config(allsize)
    
    if nargin < 1
        allsize = 12;
    end
    
    xaxis_size = allsize * 1.2;
    yaxis_size = allsize * 1.2;
    legend_size = allsize;

    ax = gca;
    ax.LineWidth = 4;
    ax.XAxis.FontSize = xaxis_size;
    ax.YAxis.FontSize = yaxis_size;
    if not(isempty(ax.Legend))
        aux = ax.Legend;
        ax.Legend.FontName = "Arial";
        ax.Legend.FontSize = legend_size;
    end
    xtickangle(0)
end