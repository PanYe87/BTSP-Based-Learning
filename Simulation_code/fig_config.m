function fig_config(allsize)
    
    if nargin < 1
        allsize = 12;
    end
    
    xaxis_size = allsize;
    yaxis_size = allsize;
    xlabel_size = round(allsize * 1.25);
    ylabel_size = round(allsize * 1.25);
    title_size = round(allsize * 1.5);
    legend_size = allsize * 0.8;

    ax = gca;
    ax.LineWidth = allsize/20;
    ax.XAxis.FontSize = xaxis_size;
    ax.YAxis.FontSize = yaxis_size;
    ax.XLabel.FontSize = xlabel_size;
    ax.YLabel.FontSize = ylabel_size;
    ax.Title.FontSize = title_size;
    ax.Title.FontName = "Arial";
    ax.XLabel.FontName = "Arial";
    ax.YLabel.FontName = "Arial";
    if not(isempty(ax.Legend))
        aux = ax.Legend;
        %ax.Legend = aux(1:2);
        ax.Legend.FontName = "Arial";
        ax.Legend.FontSize = legend_size;
    end
end