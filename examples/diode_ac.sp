Diode Small-Signal AC Analysis
* Simple diode circuit for testing AC linearization
*
* Circuit: V1 -- R1 -- D1 -- GND
*          1      2
*
* The diode is biased by DC from V1 through R1.
* AC analysis will linearize the diode at the DC operating point.

V1 1 0 DC 5 AC 1
R1 1 2 1k
D1 2 0

.AC DEC 10 1 1MEG
.END
