import streamlit as st

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


#### SEE: https://www.johndcook.com/blog/2020/01/19/cobweb-plots/
#### Plota o diagrama "cobweb"
# f = função
# x0 = ponto inicial
# N = núm. de iterações
def cobweb(f, x0, N, a=0, b=1):
        # plot the function being iterated
        t = np.linspace(a, b, 200)
        fig = plt.figure(figsize=(4,4))
        X=[]
        X.append(x0)
        #
        plt.axes().set_aspect(1)
        funcao = [f(xx) for xx in t]
        plt.plot(t, funcao, 'k')
        # plot the dotted line y = x
        plt.plot(t, t, "k:")
        # plot the INITIAL point (x0, f(x0)) => black
        plt.plot(x0, 0, 'k.')
        plt.plot( [x0,x0],[0,f(x0)], 'g')
        # plot the iterates
        x, y = x0, f(x0)
        for _ in range(N):
            fy = f(y)
            X.append(fy)
            plt.plot([x, y], [y,  y], 'g', linewidth=1)
            plt.plot([y, y], [y, fy], 'g', linewidth=1)
            plt.plot(y, fy, 'b.')
            x, y = y, fy
        plt.xlabel(r'$x_n$')
        plt.ylabel(r'$x_{n+1}$')
        plt.tight_layout()
        fig.savefig("figure1.png")
        image = Image.open('figure1.png')
        st.image(image)
        #st.pyplot(plt)
        return X

# Plota a Evolução temporal:
def Evolucao(X):
    n = len(X)
    T = np.arange(n)
    ini = 0 if n<=90 else -90
    #ymin, ymax = np.min(X[ini:])-0.2, np.max(X[ini:])+0.2
    #ymin = 0 if ymin<0 else ymin
    #ymax = 1 if ymax>1 else ymax
    fig2 = plt.figure(figsize=(5,4))
    plt.plot(T[ini:], X[ini:], '-', color='cyan')
    plt.plot(T[ini:], X[ini:], 'b.')
    plt.xlabel(r'$n$')
    plt.ylabel(r'$x_n$')
    #plt.ylim(ymin,ymax)
    plt.tight_layout()
    fig2.savefig("figure2.png")
    image2 = Image.open('figure2.png')
    st.image(image2)    

##
def Lateral(nome,eq,xi,xf,ri,rf,ID):
    st.markdown(nome)
    st.latex(eq)
    st.markdown("**Controles:**")
    r  =  st.slider(' Valor de r :', min_value=ri, max_value=rf, step=0.01, value=0.8*rf, key=ID+1)
    x0  = st.slider('Valor de x0 :', min_value=xi, max_value=xf, step=0.01, value=0.236*xf, key=ID+2)
    N  =  st.slider('Núm. máximo de iterações : ', min_value=5, max_value=2000, step=1, value=80, key=ID+3)
    return r, x0, N

############ MAPAS:
### Para cada mapa mudar: nome, eq, eq, xi, xf, ri, rf e 
### a expressão do mapa em X = cobweb(lambda x:... )

def logistico(ID):
    nome = "### Mapa Logístico"
    eq = r'''x_{n+1} = r . x_n . ( 1 - x_n )'''
    xi, xf = 0.0, 1.0 
    ri, rf = 0.0, 4.0
    col1, col2, col3, col4 = st.columns([1, 0.5, 2, 2])
    with col1:
        r, x0, N = Lateral(nome,eq,xi,xf,ri,rf,ID)
    with col2:
        st.write(" ")
    with col3:
        st.write("Mapa do 1o. retorno:")
        X = cobweb(lambda x: r*x*(1-x), x0, N)
    with col4:
        st.write("Evolução Temporal:")
        Evolucao(X)

def seno(ID):
    nome = "### Mapa do Seno"
    eq = r'''x_{n+1} = r . \sin( \pi  x_n )'''
    xi, xf = 0.0, 1.0 
    ri, rf = 0.0, 1.0
    col1, col2, col3, col4 = st.columns([1, 0.5, 2, 2])
    with col1:
        r, x0, N = Lateral(nome,eq,xi,xf,ri,rf,ID)
    with col2:
        st.write(" ")
    with col3:
        st.write("Mapa do 1o. retorno:")
        X = cobweb(lambda x: r*np.sin(np.pi*x), x0, N)
    with col4:
        st.write("Evolução Temporal:")
        Evolucao(X)
        
def tenda(ID):
    nome = "### Mapa da Tenda"
    eq = r'''
    \begin{align*}
        x_{n+1} = \left\{
        \begin{array}{cl}
        r . x     ,& x < 0.5 \\
        r . (1-x) ,& x \ge 0.5
        \end{array}
        \right.
    \end{align*}    
    '''
    xi, xf = 0.0, 1.0 
    ri, rf = 0.0, 2.0
    col1, col2, col3, col4 = st.columns([1, 0.5, 2, 2])
    with col1:
        r, x0, N = Lateral(nome,eq,xi,xf,ri,rf,ID)
    with col2:
        st.write(" ")
    with col3:
        st.write("Mapa do 1o. retorno:")
        X = cobweb(lambda x: r*x if x<0.5 else r*(1-x), x0, N)
    with col4:
        st.write("Evolução Temporal:")
        Evolucao(X)
        
############################################
## PROGRAMA PRINCIPAL:

st.set_page_config(layout="wide")

st.markdown("<h4 style='text-align: center; color: grey;'> Dinâmica de alguns Mapas Unidimensionais</h4>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["(1) Mapa Logístico ", "(2) Mapa do Seno ", " (3) Mapa da Tenda "])

####
with tab1:
    logistico(ID=10)
####
with tab2: # usar IDs diferentes p/ cada mapa
    seno(ID=20)
####
with tab3:
    tenda(ID=30)








