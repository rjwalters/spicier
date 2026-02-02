MOSFET Common-Source Amplifier AC Analysis
* Simple CS amplifier for testing MOSFET AC linearization
*
* Circuit: VDD (node 3) -- RD -- drain (node 2) -- M1 -- GND
*          Vin (node 1) -- gate
*
* DC bias: Vgs=2V, Vds set by RD

VDD 3 0 DC 10
VIN 1 0 DC 2 AC 1
RD 3 2 5k
M1 2 1 0 0 NMOS

.AC DEC 10 1 1MEG
.END
