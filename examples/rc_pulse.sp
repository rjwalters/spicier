RC Response to Pulse
* V1 pulses from 0 to 5V after 1ms delay
* Rise time 0.1ms, fall time 0.1ms, pulse width 2ms, period 5ms
V1 1 0 PULSE(0 5 1m 0.1m 0.1m 2m 5m)
R1 1 2 1k
C1 2 0 1u
.PRINT TRAN V(1) V(2)
.TRAN 0.2m 10m
.END
