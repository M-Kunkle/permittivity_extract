pla_s2p = sparameters('13.May.2023_10mm_pla.s2p');
graysla_s2p = sparameters('13.May.2023_10mm_graysla.s2p');
abs320_s2p = sparameters('13.May.2023_10mm_abs320.s2p');
hightemp_s2p = sparameters('13.May.2023_10mm_hightemp_or_clear.s2p');
air_s2p = sparameters('13.May.2023_10mm_air.s2p');
durable_s2p = sparameters('13.May.2023_10mm_durablesla.s2p');

pla_matrix = readmatrix('13.May.2023_10mm_pla_result.csv');
graysla_matrix = readmatrix('13.May.2023_10mm_graysla_result.csv');
abs320_matrix = readmatrix('13.May.2023_10mm_abs320_result.csv');
hightemp_matrix = readmatrix('13.May.2023_10mm_hightemp_or_clear_result.csv');
air_matrix = readmatrix('13.May.2023_10mm_air_result.csv');
durable_matrix = readmatrix('13.May.2023_10mm_durablesla_result.csv');

figure;
tiledlayout(1,2)
nexttile;
set(0, 'defaultfigurecolor', [1 1 1]);
set(gca, 'Fontsize', 14);
hold on;
plot(pla_s2p.Frequencies./1e9, 20*log10(abs(rfparam(pla_s2p,1,1))), 'LineWidth', 1.5, 'Color', 'blue');
plot(pla_s2p.Frequencies./1e9, 20*log10(abs(rfparam(graysla_s2p,1,1))), 'LineWidth', 1.5, 'Color', 'red');
plot(pla_s2p.Frequencies./1e9, 20*log10(abs(rfparam(abs320_s2p,1,1))), 'LineWidth', 1.5, 'Color', 'green');
plot(pla_s2p.Frequencies./1e9, 20*log10(abs(rfparam(hightemp_s2p,1,1))), 'LineWidth', 1.5, 'Color', 'yellow');
plot(pla_s2p.Frequencies./1e9, 20*log10(abs(rfparam(durable_s2p,1,1))), 'LineWidth', 1.5, 'Color', 'magenta');
%legend("PLA", "Gray SLA", "ABS320", "Hightemp or Clear(not sure which)", "Durable SLA", "Air", 'Location', 'southoutside', 'orientation', 'horizontal');
%legend('boxoff');
title("Measured De-embedded S11");
ylabel("dB");
xlabel("frequency (GHz)");
xlim([.8 2.5]);
grid on;
box on;

nexttile;
set(0, 'defaultfigurecolor', [1 1 1]);
set(gca, 'Fontsize', 14);
hold on;
plot(pla_s2p.Frequencies./1e9, real(pla_matrix(:,4)), 'LineWidth', 1.5, 'Color', 'blue');
plot(pla_s2p.Frequencies./1e9, real(graysla_matrix(:,4)), 'LineWidth', 1.5, 'Color', 'red');
plot(pla_s2p.Frequencies./1e9, real(abs320_matrix(:,4)), 'LineWidth', 1.5, 'Color', 'green');
plot(pla_s2p.Frequencies./1e9, real(hightemp_matrix(:,4)), 'LineWidth', 1.5, 'Color', 'yellow');
plot(pla_s2p.Frequencies./1e9, real(durable_matrix(:,4)), 'LineWidth', 1.5, 'Color', 'magenta');
%plot(pla_s2p.Frequencies./1e9, real(air_matrix(:,4)), 'LineWidth', 1.5, 'Color', 'black');
%legend("PLA", "Gray SLA", "ABS320", "Hightemp or Clear(not sure which)", "Durable SLA", "Air", 'Location', 'eastoutside');
%legend('boxoff');
title("Measured De-embedded Relative Permittivity");
ylabel("ε_r");
xlabel("frequency (GHz)");
xlim([.8 2.5]);
grid on;
box on;

leg = legend("PLA", "Gray SLA", "ABS320", "Hightemp or Clear(not sure which)", "Durable SLA", "Air", 'Location', 'southoutside', 'orientation', 'horizontal');
leg.Layout.Tile = 'south';