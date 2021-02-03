{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import sys\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy import optimize\n",
    "import numdifftools as nd\n",
    "from timeit import default_timer as timer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "conv_meV=27.2114*1000\n",
    "convE=51.4220674763"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "#parameters of the potential in au\n",
    "H2_shift = -1.17225263\n",
    "H2_D = 0.13681332\n",
    "H2_a = 1.21606669\n",
    "H2_re = 1.21606669\n",
    "m = 1837.47159213\n",
    "\n",
    "#cutoff to be applied on the potential, it must be equal to the one used for the exact solution\n",
    "cutoff_V=0.3\n",
    "#cutoff on the grid\n",
    "cutoff_grid=2e-16"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "#the ur-us grid\n",
    "umax=3\n",
    "_u_=np.linspace(-umax,umax,1000)\n",
    "\n",
    "ur, us=np.meshgrid(_u_,_u_)\n",
    "du_r=_u_[1]-_u_[0]\n",
    "du_s=_u_[1]-_u_[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def gen_xy(w):\n",
    "    r0,s0,XC,YC,Ur,Us,Urs=w\n",
    "    \n",
    "    r=ur+r0\n",
    "    phi=s0/r0 + 2*np.arctan2(us,2*r0)\n",
    "    \n",
    "    return XC + (ur+r0)*np.cos(phi), YC + (ur+r0)*np.sin(phi)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def _f_(w):\n",
    "    r0,s0,XC,YC,Ur,Us,Urs=w\n",
    "    \n",
    "    return np.sqrt(1+(us/(2*r0))**2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def rho(w):\n",
    "    r0,s0,XC,YC,Ur,Us,Urs=w\n",
    "    \n",
    "    U=np.array([[Ur,Urs],[Urs, Us]]) \n",
    "    \n",
    "    return du_r*du_s*np.sqrt(np.linalg.det(U/(2*np.pi)))*np.exp(-0.5*Ur*ur**2 -0.5*Us*us**2 -Urs*us*ur)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### THE POTENTIAL"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def min_V():\n",
    "\n",
    "    phimin=np.pi\n",
    "    expmin=0.5*( 1+np.sqrt(1+2*E*np.cos(phimin)/(H2_D*H2_a)) )\n",
    "    rmin=H2_re-np.log(expmin)/H2_a\n",
    "    Vmin=E*rmin*np.cos(phimin) + H2_shift + H2_D *(1 - expmin)**2\n",
    "    \n",
    "    return np.array([Vmin, rmin])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def realV(w):\n",
    "    x,y=gen_xy(w)\n",
    "    \n",
    "    rn=np.sqrt(np.log(np.exp(x**2+y**2)+cutoff_V))\n",
    "    \n",
    "    return H2_shift + H2_D*(1 - np.exp(-H2_a*(rn - H2_re)) )**2 + E*x - min_V()[0]    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def tildeV(w):\n",
    "    r0,s0,XC,YC,Ur,Us,Urs=w\n",
    "    \n",
    "    f=_f_(w)\n",
    "    r2c=np.log(np.exp((ur+r0)**2)+cutoff_grid)\n",
    "    \n",
    "    return f**2/(4*m*r2c)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### KINETIC ENERGY"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "def K_2_1(w):\n",
    "    r0,s0,XC,YC,Ur,Us,Urs=w\n",
    "    \n",
    "    f=_f_(w)\n",
    "    r2c=np.log(np.exp((ur+r0)**2)+cutoff_grid)\n",
    "           #K2                                  #K1\n",
    "    return np.array([1, f**4*r0**2/r2c])/(2*m), np.array([1/r0, us*f**2/r2c])/(2*m)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "def _K_(w):\n",
    "    r0,s0,XC,YC,Ur,Us,Urs=w\n",
    "    \n",
    "    U=np.array([[Ur,Urs],[Urs, Us]])\n",
    "    K2, K1=K_2_1(w)\n",
    "    u=ur,us\n",
    "    \n",
    "    out=0\n",
    "    for a in range(0,2):\n",
    "        out+=0.5*K2[a]*U[a,a]\n",
    "        for b in range(0,2):\n",
    "            out+=0.5*K1[a]*U[a,b]*u[b]\n",
    "            for c in range(0,2):\n",
    "                out+=-0.25*K2[a]*U[a,b]*U[a,c]*u[b]*u[c]\n",
    "    \n",
    "    return out"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "def energy(w):\n",
    "    return np.sum(rho(w)*(_K_(w) +realV(w) -tildeV(w)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### THE GRADIENT"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def forces(w):\n",
    "    x,y=gen_xy(w)\n",
    "    \n",
    "    rn=np.sqrt(np.log(np.exp(x**2+y**2)+cutoff_V))\n",
    "    drn=(rn*(1+cutoff_V*np.exp(-x**2-y**2)))**-1\n",
    "    _fx_=-2*H2_D*H2_a*(1-np.exp(-H2_a*(rn - H2_re)))*np.exp(-H2_a*(rn-H2_re))*drn*x - E\n",
    "    _fy_=-2*H2_D*H2_a*(1-np.exp(-H2_a*(rn - H2_re)))*np.exp(-H2_a*(rn-H2_re))*drn*y\n",
    "    \n",
    "    return _fx_, _fy_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "#kroneker delta\n",
    "def d(a,b):\n",
    "    if a==b:\n",
    "        return 1\n",
    "    else:\n",
    "        return 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "#OK\n",
    "def dXC(w):\n",
    "    r0,s0,XC,YC,Ur,Us,Urs=w\n",
    "    \n",
    "    fx,fy=forces(w)\n",
    "    return -np.sum(rho(w)*fx)\n",
    "\n",
    "#OK\n",
    "def dYC(w):\n",
    "    r0,s0,XC,YC,Ur,Us,Urs=w\n",
    "    \n",
    "    fx,fy=forces(w)\n",
    "    return -np.sum(rho(w)*fy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "#OK\n",
    "def ds0(w):\n",
    "    r0,s0,XC,YC,Ur,Us,Urs=w\n",
    "    fx, fy=forces(w)\n",
    "    x,y=gen_xy(w)\n",
    "    \n",
    "    return np.sum(rho(w)*(+fx*(y-YC)-fy*(x-XC)))/r0 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#OK\n",
    "def dr0(w):\n",
    "    r0,s0,XC,YC,Ur,Us,Urs=w\n",
    "    U=np.array([[Ur, Urs],[Urs, Us]])\n",
    "    \n",
    "    x,y=gen_xy(w)\n",
    "    u=ur,us\n",
    "    f=_f_(w)\n",
    "    #cutoff on the radial coordinate\n",
    "    rc=np.sqrt(np.log(np.exp((ur+r0)**2)+cutoff_grid))\n",
    "    \n",
    "    fx, fy=forces(w)\n",
    "    K2, K1=K_2_1(w) \n",
    "    \n",
    "    dK2=(f**2/rc**2)*(2*f**2*r0*ur/rc -us**2/r0)/(2*m) #OK\n",
    "    dK1=[ -1/(2*m*r0**2), -us*(f**2/rc +us**2/(4*r0**3))/(m*rc**2) ]# N0\n",
    "\n",
    "    out=0\n",
    "    for a in range(0,2):\n",
    "        out+=0.5*U[a,a]*d(a,1)*dK2\n",
    "        for b in range(0,2):\n",
    "            out+=0.5*U[a,b]*dK1[a]*u[b]\n",
    "            for c in range(0,2):\n",
    "                out+=-0.25*U[a,b]*U[a,c]*d(a,1)*dK2*u[b]*u[c]\n",
    "\n",
    "    out+=-(s0+us/f**2)*(fx*(y-YC)-fy*(x-XC))/r0**2 -(fx*(x-XC)+fy*(y-YC))/rc #OK\n",
    "    out+=+(f**2/rc +us**2/(4*r0**3))/(2*m*rc**2) #OK\n",
    "                                   \n",
    "    return np.sum(rho(w)*out)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "def dU(a,b,e,f):\n",
    "    if e!=f:\n",
    "        return d(a,e)*d(b,f)+d(b,e)*d(a,f)\n",
    "    else:\n",
    "        return d(a,e)*d(b,f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "#OK\n",
    "def dUrUs(i,j,w):\n",
    "    r0,s0,XC,YC,Ur,Us,Urs=w\n",
    "    \n",
    "    U=np.array([[Ur, Urs],[Urs, Us]])\n",
    "    Ui=np.linalg.inv(U)\n",
    "    \n",
    "    x,y=gen_xy(w)\n",
    "    u=ur,us\n",
    "    f=_f_(w)\n",
    "    rc=np.sqrt(np.log(np.exp((ur+r0)**2)+cutoff_grid))\n",
    "    \n",
    "    K2, K1=K_2_1(w) \n",
    "    fx,fy=forces(w)\n",
    "    \n",
    "    dK1=[ -f**2*us/(m*rc**3) , (f**2/rc**2 +0.5*(us/(r0*rc))**2)/(2*m) ] #OK\n",
    "    dK2=[ -f**4*r0**2/(m*rc**3), f**2*us/(2*m*rc**2) ] #OK\n",
    "    \n",
    "    \n",
    "    dtV=[ -f**2/(2*m*rc**3), us/(8*m*r0**2*rc**2) ] #OK\n",
    "    dV= [ (-fx*(x-XC)-fy*(y-YC))/rc, (fx*(y-YC)-fy*(x-XC))/(r0*f**2)] #OK\n",
    "\n",
    "    out=0\n",
    "    #OK\n",
    "    for a in range(0,2):\n",
    "        out+=+0.5*dU(a,a,i,j)*K2[a] \n",
    "        for b in range(0,2):\n",
    "            out+=+0.5*dU(a,b,i,j)*K1[a]*u[b]\n",
    "            for c in range(0,2):\n",
    "                out+=-0.5*dU(a,b,i,j)*U[a,c]*K2[a]*u[b]*u[c]\n",
    "    #OK  \n",
    "    for l in range(0,2):\n",
    "        for m_ in range(0,2):\n",
    "            for n in range(0,2):\n",
    "                temp=0 \n",
    "                for a in range(0,2):\n",
    "                    temp+=-0.25*U[a,a]*u[n]*d(a,1)*dK2[l]\n",
    "                    for b in range(0,2):\n",
    "                        temp+=-0.25*U[a,b]*u[n]*(d(a,1)*dK1[l]*u[b] +d(b,l)*K1[a])\n",
    "                        for c in range(0,2):\n",
    "                            temp+=0.125*U[a,b]*U[a,c]*u[n]*(d(a,1)*dK2[l]*u[b]*u[c] +2*K2[a]*d(b,l)*u[c])\n",
    "               \n",
    "                temp*=Ui[l,m_]*dU(m_,n,i,j)\n",
    "                out+=temp\n",
    "           \n",
    "                out+=-0.5*Ui[l,m_]*dU(m_,n,i,j)*u[n]*(dV[l]-dtV[l])  \n",
    "                \n",
    "    return np.sum(rho(w)*out)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "def grad(w):\n",
    "    return np.array([dr0(w), ds0(w), dXC(w), dYC(w), dUrUs(0,0,w), dUrUs(1,1,w), dUrUs(0,1,w)])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### MINIMIZATION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "#reading the starting values\n",
    "File='Stereographic_SCHA_GRAD_gtol1e-7_Erange1_r0_XC_Ur_Us.txt'\n",
    "start=np.array([])\n",
    "with open(File,'r') as file:\n",
    "    lines=file.read().splitlines()\n",
    "    for line in lines:\n",
    "        element=[float(eval(n)) for n in line.split()]\n",
    "        start=np.append(start,element)    \n",
    "file.close()\n",
    "start=np.reshape(start,(14,5))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.00015920968245237543"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Erange1=np.linspace(0,0.06,15)\n",
    "#Erange2=np.linspace(0,Erange1[1],6)\n",
    "i=0\n",
    "E=Erange1[i+1]\n",
    "\n",
    "#my starting parameters\n",
    "r0, s0=H2_re, np.pi*r0\n",
    "XC, YC=1, -1\n",
    "Ur, Us, Urs=45, 6, 3\n",
    "\n",
    "#from the file r0-s0-XC-YC-Ur-Us-Urs\n",
    "w0=(r0, s0, XC, YC, Ur, Us, Urs)\n",
    "w0=(start[i,0], np.pi*start[i,0], start[i,1], 0, start[i,2], start[i,3], 3)\n",
    "energy(w0)-start[i,-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-1.31044061e-03, -4.24454250e-03,  2.92232671e-03, -7.41268690e-03,\n",
       "       -6.56444137e-06,  6.33405304e-05, -3.48463926e-05])"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#nd.Gradient(energy)(w0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-6.52212455e-04,  7.77198130e-05,  3.29033532e-04,  3.32148379e-03,\n",
       "       -6.59406535e-06, -3.99531218e-05,  1.10870521e-04])"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grad(w0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "----------------------------E=4.286e-03au----------------------------\n",
      "Warning: Desired error not necessarily achieved due to precision loss.\n",
      "         Current function value: 0.007321\n",
      "         Iterations: 129\n",
      "         Function evaluations: 408\n",
      "         Gradient evaluations: 398\n",
      "199.2126026652106\n",
      "0.034769194932059946\n",
      "[-1.15621569e-05  1.06896631e-07  1.18500771e-06 -3.81768611e-09\n",
      " -2.31408228e-07  1.03674414e-07  1.25383637e-06]\n",
      "_____________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "results=np.array([])\n",
    "ZPM=np.array([])\n",
    "\n",
    "#from the file r0-s0-XC-YC-Ur-Us-Urs\n",
    "w0=(start[i,0], np.pi*start[i,0], start[i,1], 0, start[i,2], start[i,3], 3)\n",
    "for i in range(1):\n",
    "    E=Erange1[i+1]\n",
    "    \n",
    "    print('----------------------------E={:.3e}au----------------------------'.format(E))\n",
    "    \n",
    "    final=optimize.minimize(energy, x0=w0, jac=grad, method='CG', options={'gtol':1e-7, 'disp':True, 'maxiter': 4000}) \n",
    "    \n",
    "    results=np.append(results,final)\n",
    "    ZPM=np.append(ZPM,final.fun)\n",
    "    \n",
    "    print((final.fun)*conv_meV)\n",
    "    print((final.fun-start[i,-1])*conv_meV)\n",
    "    print(final.jac)\n",
    "    \n",
    "    if final.message==\"Warning: CG iterations didn't converge.  The Hessian is not positive definite.\":\n",
    "        w0=(start[i,0],np.pi*start[i,0],start[i,1],YC,start[i,2],start[i,3],Urs)\n",
    "        \n",
    "    else:\n",
    "        w0=final.x\n",
    "        \n",
    "    print('_____________________________________________________________________')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([ 1.18816875e+00, -4.20328299e-02,  4.75707724e+01,  4.50490584e+00]),\n",
       " array([ 1.20421673e+00,  3.70276350e+00, -2.76206004e-02, -8.42251075e-02,\n",
       "         4.74446337e+01,  4.70626032e+00,  2.69310364e+00]))"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "start[0,:-1],final.x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "createfile=False\n",
    "if createfile==True:\n",
    "    File='Stereographic_SCHA_GRAD_gtol1e-7_Erange1_down_r0_XC_Ur_Us.txt'\n",
    "    with open(File,'w') as f:\n",
    "        for i in range(np.sum(np.shape(results))):\n",
    "            for j in range(np.sum(np.shape(results[0].x))):\n",
    "                f.write('{} '.format(np.flip(results)[i].x[j]) ) \n",
    "            f.write('{} '.format(np.flip(results)[i].fun) ) \n",
    "            f.write('\\n')\n",
    "    f.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
