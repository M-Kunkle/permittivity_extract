pla_s2p = sparameters('13.May.2023_10mm_pla.s2p');
graysla_s2p = sparameters('13.May.2023_10mm_graysla.s2p');
abs320_s2p = sparameters('13.May.2023_10mm_abs320.s2p');
hightemp_s2p = sparameters('13.May.2023_10mm_hightemp_or_clear.s2p');
air_s2p = sparameters('13.May.2023_10mm_air.s2p');
durable_s2p = sparameters('13.May.2023_10mm_durablesla.s2p');

pla_matrix = readmatrix('13.May.2023_10mm_pla_result.csv');

figure;
set(0, 'defaultfigurecolor', [1 1 1]);
set(gca, 'Fontsize', 14);
hold on;
plot(pla_s2p.Frequencies./1e9, 20*log10(abs(rfparam(pla_s2p,1,1))), 'LineWidth', 1.5, 'Color', 'blue');
plot(pla_s2p.Frequencies./1e9, 20*log10(abs(rfparam(graysla_s2p,1,1))), 'LineWidth', 1.5, 'Color', 'red');
plot(pla_s2p.Frequencies./1e9, 20*log10(abs(rfparam(abs320_s2p,1,1))), 'LineWidth', 1.5, 'Color', 'green');
plot(pla_s2p.Frequencies./1e9, 20*log10(abs(rfparam(hightemp_s2p,1,1))), 'LineWidth', 1.5, 'Color', 'yellow');
%plot(pla_s2p.Frequencies, 20*log10(abs(rfparam(air_s2p,1,1))), 'LineWidth', 1.5, 'Color', 'cyan');
plot(pla_s2p.Frequencies./1e9, 20*log10(abs(rfparam(durable_s2p,1,1))), 'LineWidth', 1.5, 'Color', 'magenta');
legend("pla", "gray sla", "abs320", "hightemp or clear(not sure which)", "durable sla");
title("Measured De-embedded S11");
ylabel("dB");
xlabel("frequency (GHz)");
xlim([.8 2.5]);

grid on;
box on;

figure;
set(0, 'defaultfigurecolor', [1 1 1]);
set(gca, 'Fontsize', 14);
hold on;
plot(pla_s2p.Frequencies./1e9, 20*log10(abs(rfparam(pla_s2p,1,1))), 'LineWidth', 1.5, 'Color', 'blue');
plot(pla_s2p.Frequencies./1e9, 20*log10(abs(rfparam(graysla_s2p,1,1))), 'LineWidth', 1.5, 'Color', 'red');
plot(pla_s2p.Frequencies./1e9, 20*log10(abs(rfparam(abs320_s2p,1,1))), 'LineWidth', 1.5, 'Color', 'green');
plot(pla_s2p.Frequencies./1e9, 20*log10(abs(rfparam(hightemp_s2p,1,1))), 'LineWidth', 1.5, 'Color', 'yellow');
%plot(pla_s2p.Frequencies, 20*log10(abs(rfparam(air_s2p,1,1))), 'LineWidth', 1.5, 'Color', 'cyan');
plot(pla_s2p.Frequencies./1e9, 20*log10(abs(rfparam(durable_s2p,1,1))), 'LineWidth', 1.5, 'Color', 'magenta');
legend("pla", "gray sla", "abs320", "hightemp or clear(not sure which)", "durable sla");
title("Measured De-embedded S11");
ylabel("dB");
xlabel("frequency (GHz)");
xlim([.8 2.5]);

grid on;
box on;