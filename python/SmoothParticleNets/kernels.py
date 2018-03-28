
KERNELS = {}
DKERNELS = {}

""" POLY:
	\eta * \sigma * (1/4*max(0, H - d)^3 - max(0, H/2 - d)^3)
		 H = radius
		 d = distance
	\sigma = 1/pi (dim norm)
	  \eta = 8*pi/(H^3) (norm)
"""
KERNELS["poly"] = ("8.0f/(H*H*H)*((H-d)*(H-d)*(H-d)/4.0f - "
	"fmaxf(0.0f, H/2.0f-d)*(H/2.0f-d)*(H/2.0f-d))")

""" DPOLY:
	3 * \eta * \sigma * (-1/4*max(0, H - d)^2 + max(0, H/2 - d)^2)
		 H = radius
		 d = distance
	\sigma = 1/pi (dim norm)
	  \eta = 8*pi/(H^3) (norm)
"""
KERNELS["dpoly"] = ("24.0f/(H*H*H)*(-(H-d)*(H-d)/4.0f + "
	"fmaxf(0.0f, H/2.0f-d)*(H/2.0f-d))")
DKERNELS["poly"] = KERNELS["dpoly"]

""" DPOLY2:
	6 * \eta * \sigma * (1/4*max(0, H - d) - max(0, H/2 - d))
		 H = radius
		 d = distance
	\sigma = 1/pi (dim norm)
	  \eta = 8*pi/(H^3) (norm)
"""
KERNELS["dpoly2"] = "42.0f/(H*H*H)*((H-d)/4.0f + fmaxf(0.0f, H/2.0f-d))"
DKERNELS["dpoly"] = KERNELS["dpoly2"]
DKERNELS["dpoly2"] = "42.0f/(H*H*H)*(3.0f/4.0f)"

""" DEFAULT:
	\eta * \sigma * max(0, H^2 - d^2)^3
		 H = radius
		 d = distance
	\sigma = 1/pi (dim norm)
	  \eta = 315/(64*H^9) (norm)
"""
KERNELS["default"] = ( 
	"(315.0f/(64.0f*M_PI*H*H*H*H*H*H*H*H*H))*(H*H-d*d)*(H*H-d*d)*(H*H-d*d)")

""" DDEFAULT:
	\eta * \sigma * d * max(0, H^2 - d^2)^2
		 H = radius
		 d = distance
	\sigma = 1/pi (dim norm)
	  \eta = -945/(32*H^9) (norm)
"""
KERNELS["ddefault"] = "(-945.0f/(32.0f*M_PI*H*H*H*H*H*H*H*H*H))*(H*H-d*d)*(H*H-d*d)*d"
DKERNELS["default"] = KERNELS["ddefault"]

""" DDEFAULT2:
	\eta * \sigma * (H^4 - 6*H^2*d^2 + 5d^4)
		 H = radius
		 d = distance
	\sigma = 1/pi (dim norm)
	  \eta = -945/(32*H^9) (norm)
"""
KERNELS["ddefault2"] = "(-945.0f/(32.0f*M_PI*H*H*H*H*H*H*H*H*H))*(H*H*H*H - 6*H*H*d*d + 5*d*d*d*d)"
DKERNELS["ddefault"] = KERNELS["ddefault2"]
DKERNELS["ddefault2"] = "(-945.0f/(32.0f*M_PI*H*H*H*H*H*H*H*H*H))*(20*d*d*d - 12*H*H*d)"

""" PRESSURE:
	\eta * \sigma * max(0, H - d)^3
		 H = radius
		 d = distance
	\sigma = 1/pi (dim norm)
	  \eta = 15/(H^6) (norm)
"""
KERNELS["pressure"] = "(15.0f/(M_PI*H*H*H*H*H*H))*(H-d)*(H-d)*(H-d)"

""" DPRESSURE:
	\eta * \sigma * max(0, H - d)^2
		 H = radius
		 d = distance
	\sigma = 1/pi (dim norm)
	  \eta = -45/(H^6) (norm)
"""
KERNELS["dpressure"] = "(-45.0f/(M_PI*H*H*H*H*H*H))*(H-d)*(H-d)"
DKERNELS["pressure"] = KERNELS["dpressure"]

""" DPRESSURE2:
	\eta * \sigma * max(0, H - d) * (H - 2*d)/2
		 H = radius
		 d = distance
	\sigma = 1/pi (dim norm)
	  \eta = -90/(H^6) (norm)
"""
KERNELS["dpressure2"] = "(-90.0f/(M_PI*H*H*H*H*H*H))*(H-d)*(H-2*d)/d"
DKERNELS["dpressure"] = KERNELS["dpressure2"]
DKERNELS["dpressure2"] = "(-90.0f/(M_PI*H*H*H*H*H*H))*(2.0f*d - 3.0f*H/2.0f)"

""" VISCOSITY:
	\eta * \sigma * (-d^3/(2*H^3) + d^2/(H^2) + H/(2*d) - 1)
		 H = radius
		 d = distance
	\sigma = 1/pi (dim norm)
	  \eta = 15/(2*H^3) (norm)
"""
KERNELS["viscosity"] = "(15.0f/(2*M_PI*H*H*H))*(-d*d*d/(2*H*H*H)+d*d/(H*H)+H/(2*(d+1e-15))-1)"

""" DVISCOSITY:
	\eta * \sigma * d * (-3*d/(2*H^3) + 2/(H^2) - H/(2*d^3))
		 H = radius
		 d = distance
	\sigma = 1/pi (dim norm)
	  \eta = 15/(2*H^3) (norm)
"""
KERNELS["dviscosity"] = (
	"(15.0f/(2*M_PI*H*H*H))*d*(-3*d/(2*H*H*H)+2/(H*H)-H/(2*(d+1e-15)*(d+1e-15)*(d+1e-15)))")
DKERNELS["viscosity"] = KERNELS["dviscosity"]

""" DVISCOSITY2:
	\eta * \sigma * max(0, H - d)
		 H = radius
		 d = distance
	\sigma = 1/pi (dim norm)
	  \eta = 45/(H^6) (norm)
"""
KERNELS["dviscosity2"] = "(45.0f/(M_PI*H*H*H*H*H*H))*(H-d)"
DKERNELS["dviscosity"] = KERNELS["dviscosity2"]
DKERNELS["dviscosity2"] = "(45.0f/(M_PI*H*H*H*H*H*H))*-1.0f"

""" INDIRECT:
	H - d
		 H = radius
		 d = distance
"""
KERNELS["indirect"] = "H - d"
DKERNELS["indirect"] = "-1.0f"

""" CONSTANT:
	1
"""
KERNELS["constant"] = "1.0f"
DKERNELS["constant"] = "0.0f"

""" SPIKY:
	\eta * \sigma * (1 - d/H)^2
		 H = radius
		 d = distance
	\sigma = 1/pi (dim norm)
	  \eta = 15/(H^3) (norm)
"""
KERNELS["spiky"] = "15.0f/(M_PI*H*H*H)*(1.0f-d/H)*(1.0f-d/H)"

""" DSPIKY:
	\eta * \sigma * 2 * (1 - d/H)/H
		 H = radius
		 d = distance
	\sigma = 1/pi (dim norm)
	  \eta = 15/(H^3) (norm)
"""
KERNELS["dspiky"] = "-15.0f/(M_PI*H*H*H)*2.0f*(1.0f - d/H)/H"
DKERNELS["spiky"] = KERNELS["dspiky"]
DKERNELS["dspiky"] = "-15.0f/(M_PI*H*H*H)*2.0f*(-1.0f/H)/H"

""" COHESION:
-(1.0f + \eta)/\eta^2*(d/H)^3 + (\eta^2 + \eta + 1)/\eta^2*(d/H)^2 - 1
	\eta * \sigma * (1 - d/H)^2
		 H = radius
		 d = distance
	  \eta = 0.5 (rest)
"""
KERNELS["cohesion"] = "-6.0f*(d/H)*(d/H)*(d/H) + 7*(d/H)*(d/H) - 1"
DKERNELS["cohesion"] = "2.0f*d*(7.0f*H - 9.0f*d)/(H*H*H)"

KERNEL_NAMES = sorted(KERNELS.keys())