from flask import Flask, render_template,  request
from flask import send_file
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import mplstereonet
import numpy as np
import math

app = Flask(__name__)
app.debug = True 
@app.route("/")
def func():
    return render_template('index.html')
@app.route('/download-image')
def download_image():
    image_path = './my_image.png'
    return send_file(image_path, as_attachment=True)
@app.route('/download-new-image')
def download_new_image():
    image_path = './my_image2.png'
    return send_file(image_path, as_attachment=True)
@app.route("/plotstereonet" , methods=['POST','GET'])
def func1():
  np.float = float
  my_value=0
  plot_url1=''
  if request.method=='POST':
     
    slope_face_strike=request.form['slope_face_strike']
    slope_face_dip=request.form['slope_face_dip']
    if slope_face_strike:
     slope_face_strike = int(slope_face_strike)
    if slope_face_dip:
     slope_face_dip = int(slope_face_dip)
    strike=[]
    dip=[]
    if len(request.form['strike1'])>0:
     strike.append(int(request.form['strike1']))
    if len(request.form['strike2'])>0:
     strike.append(int(request.form['strike2']))
    if len(request.form['strike3'])>0:
     strike.append(int(request.form['strike3']))
    if len(request.form['dip1'])>0:
     dip.append(int(request.form['dip1']))
    if len(request.form['dip2'])>0: 
     dip.append(int(request.form['dip2']))
    if len(request.form['dip3'])>0:
     dip.append(int(request.form['dip3']))
     # create the stereonet plot
    color ='gbcmykw'
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='stereonet')
   
    for i in range(len(dip)):
     if abs(strike[i]-slope_face_strike)<=20 and abs(dip[i]-slope_face_dip)<20:
      print("Planer Failure plane 1 :  Strike - " , strike[i] , " dip -" , dip[i])
     ax.plane(strike[i], dip[i], c =color[i%7], label='plane %1d =  %03d/%02d' % (i+1 , strike[i], dip[i]))
     ax.pole(strike[i], dip[i], c=color[i%7], label='pole of plane %1d '%(i+1))
    ax.plane(slope_face_strike, slope_face_dip, c='r', label='Fault %03d/%02d' % (slope_face_strike, slope_face_dip))
    ax.legend()
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    with open('my_image.png', 'wb') as f:
      f.write(img.getbuffer())
    
    plot_url = base64.b64encode(img.getvalue()).decode()
    return render_template('plot_stereonet.html' , my_value=my_value, plot_url=plot_url, plot_url1=plot_url1)
  else :
    return render_template('plot_stereonet.html')   

@app.route("/failure-analysis" , methods=['POST','GET'])
def failure():

    # Wedge Analysis 
  np.float = float
    # Inputs 
    #  Dip and Strike of planes
  if request.method=='POST':
  
     s1 , a1 = int(request.form['dip1']) , int(request.form['strike1'])  #Plane 1
     s2 , a2 = int(request.form['dip2']) , int(request.form['strike2'])   #Plane 2
     s3 , a3 = int(request.form['dip3']) , int(request.form['strike3'])    #Plane 3
     s4 , a4 =int(request.form['dip4']) , int(request.form['strike4'])   #Plane 4
     s5 , a5 = int(request.form['dip5']) , int(request.form['strike5'])   #Plane 5 
     face_dip=s1
     dips=[]
     strikes=[ ]
     dips.append(s2)
     dips.append(s3)
     dips.append(s4)
     dips.append(s5)
     face_strike=a1
     strikes.append(a2)
     strikes.append(a3)
     strikes.append(a4)
     strikes.append(a5)

     sT , aT =int(request.form['sT']) , int(request.form['aT'] )#Anchor Tension
     sE , aE = int(request.form['sE']) , int(request.form['aE'] )  #External Load
     T =int(request.form['T']) 
     E = int(request.form['E'] )
     phi_1 = int(request.form['phi1'])
     phi_2 =int(request.form['phi2'])
     c1 = int(request.form['c1'])
     c2 = int(request.form['c2'])

     H1 = int(request.form['H1']  )    #Height of Plane1
     L = int(request.form['L'] )       #Length of distance of crack from crest , along Plane1
     ita = int(request.form['itta'])
     gamma_rock = int(request.form['gamma-rock'])
     gamma_water =int(request.form['gama-water'])
    #  Wedge Failure Analysis Through Equation and Given Inputs
     color ='gbcmykw'
     fig1= plt.figure(figsize=(8,8))
     ax = fig1.add_subplot(111, projection='stereonet')
     for i in range(len(strikes)):
      ax.plane(strikes[i], dips[i], c =color[i%7], label='plane %1d =  %03d/%02d' % (i+1 , strikes[i], dips[i]))
      ax.pole(strikes[i], dips[i], c=color[i%7], label='pole of plane %1d '%(i+1))
     for j in range(i+1 ,len(strikes)):
      plunge, trend = mplstereonet.plane_intersection(strikes[i], dips[i], strikes[j], dips[j])
      if face_dip > plunge and plunge > friction:
        if trend>face_strike and trend<face_strike+180:
         print("Wedge Failure possible Planes " , i+1 , " and " , j+1)
      ax.line(plunge, trend,c=color[j%7], markersize=5, 
         label='Intersection %02d/%03d' % (plunge, trend))
     angle = mplstereonet.angular_distance((12, 95), (235, 70) ,bidirectional=True)
     print(np.degrees(angle))
     ax.plane(face_strike, face_dip, c='r', label='Fault %03d/%02d' % (face_strike, face_dip))
     ax.legend()
     ax.grid()

     # convert the plot to a base64 encoded image
     img = BytesIO()
     fig1.savefig(img, format='png')
     img.seek(0)
     with open('my_image2.png', 'wb') as f:
      f.write(img.getbuffer())
     plot_url = base64.b64encode(img.getvalue()).decode()

     #Calculation of Normal Vector - s(i) is strike and a(i) is dip of the five planes

     ax , ay , az = math.sin(math.radians(s1))*math.sin(math.radians(a1)) , math.sin(math.radians(s1))*math.cos(math.radians(a1)) , math.cos(math.radians(s1))
     bx , by , bz = math.sin(math.radians(s2))*math.sin(math.radians(a2)) , math.sin(math.radians(s2))*math.cos(math.radians(a2)) , math.cos(math.radians(s2))
     dx , dy , dz = math.sin(math.radians(s3))*math.sin(math.radians(a3)) , math.sin(math.radians(s3))*math.cos(math.radians(a3)) , math.cos(math.radians(s3))
     fx , fy , fz = math.sin(math.radians(s4))*math.sin(math.radians(a4)) , math.sin(math.radians(s4))*math.cos(math.radians(a4)) , math.cos(math.radians(s4))
     f5x , f5y , f5z = math.sin(math.radians(s5))*math.sin(math.radians(a5)) , math.sin(math.radians(s5))*math.cos(math.radians(a5)) , math.cos(math.radians(s5))
     tx , ty , tz = math.cos(math.radians(sT))*math.sin(math.radians(aT)) ,math.cos(math.radians(sT))*math.cos(math.radians(aT)) , -math.sin(math.radians(sT))
     ex , ey , ez = math.cos(math.radians(sE))*math.sin(math.radians(aE)) ,math.cos(math.radians(sE))*math.cos(math.radians(aE)) , -math.sin(math.radians(sE))

     #Vectors in the Direction of the line of Intersection of Various planes
     gx , gy , gz = ( fy*az - fz*ay ) , ( fz*az - fx*az ) , ( fx*ay - fy*ax )          #Intersection of Plane 1 & Plane 4
     g5x , g5y , g5z = ( f5y*az - f5z*ay )  , ( f5z*ax -f5x*az ) , ( f5x*ay - f5y*ax)  #Intersection of Plane 1 & Plane 5
     ix , iy , iz = ( by*az - bz*ay ) , ( bz*ax - bx*az ) , ( bx*ay - by*ax )          #Intersection of Plane 1 & Plane 2
     jx , jy , jz = ( fy*dz - fz*dy ) , ( fz*dx - fx*dz ) , ( fx*dy - fy*dx )          #Intersection of Plane 3 & Plane 4
     j5x , j5y , j5z = ( f5y*dz - f5z*dy ) , ( f5z*dx - f5x*dz ) , ( f5x*dy - f5y*dx ) #Intersection of Plane 3 & Plane 5
     kx , ky , kz = ( iy*bz - iz*by ) , ( iz*bx - ix*bz ) , ( ix*by - iy*bx )          #Vector in Plane 2 & Normal to i vector(Intersection of plane 1 & Plane 2)
     lx , ly , lz = ( ay*iz - az*iy ) , ( az*ix - ax*iz ) , ( ax*iy - ay*ix)           #Vector in Plane 1 Normal to vector i(Intersection of plane 1 & Plane 2)
         
           #Number Proportional to Cosine of Various Angles
     m = gx*dx + gy*dy + gz*dz
     m5 = g5x*dx +  g5y*dy + g5z*dz
     n = bx*jx + by*jy +bz*jz
     n5 = bx*j5x + by*j5y + bz*j5z
     p = ix*dx + iy*dy + iz*dz
     q = bx*gx + by*gy + bz*gz
     g5 = bx*g5x + by*g5y + bz*g5z
     r = ax*bx + ay*by + az*bz
     s = ax*tx + ay*ty + az*tz
     u = bx*tx + by*ty + bz*tz
     w = ix*tx + iy*ty + iz*tz 
     se = ax*ex +ay*ey + az*ez
     ue = bx*ex +by*ey + bz*ez
     we = ix*ex +iy*ey + iz*ez
     s5 = ax*f5x + ay*f5y + az*f5z
     v5 = bx*f5x + by*f5y + bz*f5z
     w5 = ix*f5x + iy*f5y + iz*f5z
     l = ix*gx + iy*gy + iz*gz
     l5 =  ix*g5x + iy*g5y + iz*g5z
     e = fx*f5x + fy*f5y + fz*f5z

     q5 = bx*g5x + by*g5y + bz*g5z
   
     # Miscellaneous Factors

     R = math.sqrt(1 - r*r)
     rho = 1/(R*R)*(n*q)/abs(n*q)
     meu = 1/(R*R)*(m*q)/abs(m*q)
     v = 1/R*(p/abs(p))
     G = gx*gx + gy*gy + gz*gz
     G5 = g5x*g5x + g5y*g5y + g5z*g5z
     M = math.sqrt(G*p*p - 2*m*p*l + m*m*R*R)
     M5 = math.sqrt(G5*p*p - 2*m5*p*l5 + m5*m5*R*R)
     h = H1/abs(gz)
     h5 = (M*h - abs(p)*L)/M5
     B = (math.tan(math.radians(phi_1))*math.tan(math.radians(phi_1)) + math.tan(math.radians(phi_2))*math.tan(math.radians(phi_2)) - 2*(meu*r/rho)*math.tan(math.radians(phi_1))*math.tan(math.radians(phi_2))) / (R*R)
     si = math.asin(v*iz)*57.29578
     ai = math.atan((-v*ix)/(-v*iy))*57.29578

     if ai<min(a1,a2):
       ai+=180
     elif ai>max(a1,a2):
       ai-=180
    
    # Check on Wedge Geometry
     if p*iz < 0 or n*q*iz < 0 :
       return render_template("failure_analysis.html" , ans="NO Wedge is Formed " , plot_url=plot_url)
       print("NO Wedge is Formed ")

     if(e*ita*q5*iz < 0 or h5<0 or abs((m5*h5)/(m*h))>1 or abs((n*q5*m5*h5)/(n5*q*m*h))>1):
       return render_template("failure_analysis.html" , ans1="Tension Crack Invalid",  plot_url=plot_url )
       print("Tension Crack Invalid")
       print(p*iz ,n*q*iz)

     A1 = (abs(m*q)*h*h - abs(m5*q5)*h5*h5)/(2*abs(p))
     A2 = ((abs(q)*m*m*h*h)/abs(n) - (abs(q5)*m5*m5*h5*h5)/abs(n5))/(2*abs(p))
     A5 = (abs(m5*q5)*h5*h5)/(2*n5)
     W = gamma_rock*((q*q*m*m*h*h*h/abs(n) - q5*q5*m5*m5*h5*h5*h5/abs(n5))/abs(6*p))
     print(A1 , A2 ,A5 , W)

     #Water Pressure
     # 1. With tension Crack
     u1 = (gamma_water*h5*abs(m5))/(3*dz)
     u2 = u1
     u5 = u1
     V = u5*A5*ita*(e/abs(e))
     print(u1 , V)

     #2. Without Tention Crack
     u1 = (gamma_water*h5*abs(m5))/(3*dz)
     u2 = u1
     u5 = u1
     V = u5*A5*ita*(e/abs(e))
     print(u1 , V)

     #Normal Reactions on Plane 1 and Plane 2 Contact on both planes
     N1 = rho*(W*kz + T*(r*v - s) + E*(e*ue - se) + (V*(r*v5 - s5))) - u1*A1
     N2 = meu*(W*lz + T*(r*s - v) + E*(r*se -ue) + V*(r*s5 - v5)) - u2*A2
     print(N1 , N2)
     if(N1 < 0 and N2<0):
       
       print("Factor of Safety is Zero ")

     if(N1 > 0 and N2 < 0):
       Na = W*az - T*s - E*se - V*s5 - u1*A1*r
       Sx = T*tx + E*ex + Na*ax + V*f5x + u1*A1*bx
       Sy = T*ty + E*ey + Na*ay + V*f5y + u1*A1*by
       Sz = (T*tz + E*ez + Na*az + V*f5z + u1*A1*bz) + W
       Sa = math.sqrt(Sx*Sx + Sy*Sy + Sz*Sz)
       Qa = (Na - u1*A1)*math.tan(phi_1) + c1*A1
       Factor_of_safety_1 = (Qa / Sa)
       print(Factor_of_safety_1)
       return render_template("failure_analysis.html" , fos=Factor_of_safety_1 ,  plot_url=plot_url , w=W , a1=A1 , a2=A2 , n1=N1, n2=N2)
     if(N1 < 0 and N2 > 0):
       Nb = (W*bz - T*u - E*ue - u2*A2*r)
       Sx = (T*tx + E*ex + Nb*bx + V*f5x + u2*A2*ax)
       Sy = (T*ty + E*ey + Nb*by + V*f5y + u2*A2*ay)
       Sz = (T*tz + E*ez + Nb*bz + V*f5z + u2*A2*az) + W
       Sb = math.sqrt(Sx*Sx + Sy*Sy + Sz*Sz)
       Qb = (Nb - u2*A2)*math.tan(phi_2) + c2*A2
       Factor_of_safety_2 = (Qb/Sb)
       return render_template("failure_analysis.html" , fos=Factor_of_safety_2 ,  plot_url=plot_url , w=W , a1=A1 , a2=A2 , n1=N1, n2=N2)
       print(Factor_of_safety_2)
     if(N1 > 0 and N2 > 0):
       S = v*(W*iz - T*w - E*we - V*w5)
       Q = N1*math.tan(math.radians(phi_1)) + N2*math.tan(math.radians(phi_2)) + c1*A1 + c2*A2
       Factor_of_safety_3 = (Q/S)
       return render_template("failure_analysis.html" , fos=Factor_of_safety_3 ,  plot_url=plot_url , w=W , a1=A1 , a2=A2 , n1=N1, n2=N2)
       # print(S , Q ,iz , w5 ,v)
      
     # render the HTML template and pass in the base64 encoded image
  return render_template("failure_analysis.html"  )
@app.route("/app")
def home():
    # # create the stereonet plot
    # np.float = float
    # # my_array = np.zeros((10,), dtype=np.float)
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='stereonet')
    # # strike=[210,310,200,110]
    # # dip=[20,30,40,90]
    # # ax.plane(strike, dip, 'g-', linewidth=2)
    # # ax.pole(strike, dip, 'g^', markersize=18) 
    # # ax.rake(strike, dip, -25)
    # # ax.grid()
    # slope_face_strike = 30
    # slope_face_dip = 40
    # strike = [ 22, 265 , 213]
    # dip = [65 , 68 , 55]
    # fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(111, projection='stereonet')
    # for i in range(3):
    #  if abs(strike[i]-slope_face_strike)<=20 and abs(dip[i]-slope_face_dip)<20:
    #   print("Planer Failure plane 1 :  Strike - " , strike[i] , " dip -" , dip[i])
    #  ax.plane(strike[i], dip[i], c='b', label='Bedding %03d/%02d' % (strike[i], dip[i]))
    #  ax.pole(strike[i], dip[i], c='r', label='Beta axis (Intersection of Planes)')
    # ax.plane(slope_face_strike, slope_face_dip, c='r', label='Fault %03d/%02d' % (slope_face_strike, slope_face_dip))
    # ax.legend()
    # img = BytesIO()
    # fig.savefig(img, format='png')
    # img.seek(0)
    # plot_url1 = base64.b64encode(img.getvalue()).decode()
    
    # # Wedge Analysis 
    # face_dip = 56
    # face_strike = 342
    # strikes = [ 22, 265 , 213]
    # dips = [65 , 68 , 55]
    # friction = 25
    # color ='gbcmykw'
    # fig1= plt.figure(figsize=(8,8))
    # ax = fig1.add_subplot(111, projection='stereonet')
    # for i in range(len(strikes)):
    #  ax.plane(strikes[i], dips[i], c =color[i%7], label='plane %1d =  %03d/%02d' % (i+1 , strikes[i], dips[i]))
    #  ax.pole(strikes[i], dips[i], c=color[i%7], label='pole of plane %1d '%(i+1))
    # for j in range(i+1 ,len(strikes)):
    #  plunge, trend = mplstereonet.plane_intersection(strikes[i], dips[i], strikes[j], dips[j])
    #  if face_dip > plunge and plunge > friction:
    #    if trend>face_strike and trend<face_strike+180:
    #     print("Wedge Failure possible Planes " , i+1 , " and " , j+1)
    #  ax.line(plunge, trend,c=color[j%7], markersize=5, 
    #     label='Intersection %02d/%03d' % (plunge, trend))
    # angle = mplstereonet.angular_distance((12, 95), (235, 70) ,bidirectional=True)
    # print(np.degrees(angle))
    # ax.plane(face_strike, face_dip, c='r', label='Fault %03d/%02d' % (face_strike, face_dip))
    # ax.legend()
    # ax.grid()

    # # convert the plot to a base64 encoded image
    # img = BytesIO()
    # fig1.savefig(img, format='png')
    # img.seek(0)
    # plot_url = base64.b64encode(img.getvalue()).decode()

    # # Inputs 
    # #  Dip and Strike of planes
    # s1 , a1 = 45 , 105   #Plane 1
    # s2 , a2 = 70 , 235   #Plane 2
    # s3 , a3 =  12 , 195    #Plane 3
    # s4 , a4 = 65 , 185   #Plane 4
    # s5 , a5 = 70 , 165   #Plane 5 

    # sT , aT = 0 , 0  #Anchor Tension
    # sE , aE = 0 , 0  #External Load
    # T = 0 
    # E = 0
    # phi_1 = 20
    # phi_2 = 30
    # c1 = 500
    # c2 = 1000

    # H1 = 100        #Height of Plane1
    # L = 40         #Length of distance of crack from crest , along Plane1
    # ita = 1
    # gamma_rock = 160
    # gamma_water = 62.4
    # #Wedge Failure Analysis Through Equation and Given Inputs


    # #Calculation of Normal Vector - s(i) is strike and a(i) is dip of the five planes

    # ax , ay , az = math.sin(math.radians(s1))*math.sin(math.radians(a1)) , math.sin(math.radians(s1))*math.cos(math.radians(a1)) , math.cos(math.radians(s1))
    # bx , by , bz = math.sin(math.radians(s2))*math.sin(math.radians(a2)) , math.sin(math.radians(s2))*math.cos(math.radians(a2)) , math.cos(math.radians(s2))
    # dx , dy , dz = math.sin(math.radians(s3))*math.sin(math.radians(a3)) , math.sin(math.radians(s3))*math.cos(math.radians(a3)) , math.cos(math.radians(s3))
    # fx , fy , fz = math.sin(math.radians(s4))*math.sin(math.radians(a4)) , math.sin(math.radians(s4))*math.cos(math.radians(a4)) , math.cos(math.radians(s4))
    # f5x , f5y , f5z = math.sin(math.radians(s5))*math.sin(math.radians(a5)) , math.sin(math.radians(s5))*math.cos(math.radians(a5)) , math.cos(math.radians(s5))
    # tx , ty , tz = math.cos(math.radians(sT))*math.sin(math.radians(aT)) ,math.cos(math.radians(sT))*math.cos(math.radians(aT)) , -math.sin(math.radians(sT))
    # ex , ey , ez = math.cos(math.radians(sE))*math.sin(math.radians(aE)) ,math.cos(math.radians(sE))*math.cos(math.radians(aE)) , -math.sin(math.radians(sE))

    # #Vectors in the Direction of the line of Intersection of Various planes
    # gx , gy , gz = ( fy*az - fz*ay ) , ( fz*az - fx*az ) , ( fx*ay - fy*ax )          #Intersection of Plane 1 & Plane 4
    # g5x , g5y , g5z = ( f5y*az - f5z*ay )  , ( f5z*ax -f5x*az ) , ( f5x*ay - f5y*ax)  #Intersection of Plane 1 & Plane 5
    # ix , iy , iz = ( by*az - bz*ay ) , ( bz*ax - bx*az ) , ( bx*ay - by*ax )          #Intersection of Plane 1 & Plane 2
    # jx , jy , jz = ( fy*dz - fz*dy ) , ( fz*dx - fx*dz ) , ( fx*dy - fy*dx )          #Intersection of Plane 3 & Plane 4
    # j5x , j5y , j5z = ( f5y*dz - f5z*dy ) , ( f5z*dx - f5x*dz ) , ( f5x*dy - f5y*dx ) #Intersection of Plane 3 & Plane 5
    # kx , ky , kz = ( iy*bz - iz*by ) , ( iz*bx - ix*bz ) , ( ix*by - iy*bx )          #Vector in Plane 2 & Normal to i vector(Intersection of plane 1 & Plane 2)
    # lx , ly , lz = ( ay*iz - az*iy ) , ( az*ix - ax*iz ) , ( ax*iy - ay*ix)           #Vector in Plane 1 Normal to vector i(Intersection of plane 1 & Plane 2)
         
    #      #Number Proportional to Cosine of Various Angles
    # m = gx*dx + gy*dy + gz*dz
    # m5 = g5x*dx +  g5y*dy + g5z*dz
    # n = bx*jx + by*jy +bz*jz
    # n5 = bx*j5x + by*j5y + bz*j5z
    # p = ix*dx + iy*dy + iz*dz
    # q = bx*gx + by*gy + bz*gz
    # g5 = bx*g5x + by*g5y + bz*g5z
    # r = ax*bx + ay*by + az*bz
    # s = ax*tx + ay*ty + az*tz
    # u = bx*tx + by*ty + bz*tz
    # w = ix*tx + iy*ty + iz*tz 
    # se = ax*ex +ay*ey + az*ez
    # ue = bx*ex +by*ey + bz*ez
    # we = ix*ex +iy*ey + iz*ez
    # s5 = ax*f5x + ay*f5y + az*f5z
    # v5 = bx*f5x + by*f5y + bz*f5z
    # w5 = ix*f5x + iy*f5y + iz*f5z
    # l = ix*gx + iy*gy + iz*gz
    # l5 =  ix*g5x + iy*g5y + iz*g5z
    # e = fx*f5x + fy*f5y + fz*f5z

    # q5 = bx*g5x + by*g5y + bz*g5z
   
    # # Miscellaneous Factors

    # R = math.sqrt(1 - r*r)
    # rho = 1/(R*R)*(n*q)/abs(n*q)
    # meu = 1/(R*R)*(m*q)/abs(m*q)
    # v = 1/R*(p/abs(p))
    # G = gx*gx + gy*gy + gz*gz
    # G5 = g5x*g5x + g5y*g5y + g5z*g5z
    # M = math.sqrt(G*p*p - 2*m*p*l + m*m*R*R)
    # M5 = math.sqrt(G5*p*p - 2*m5*p*l5 + m5*m5*R*R)
    # h = H1/abs(gz)
    # h5 = (M*h - abs(p)*L)/M5
    # B = (math.tan(math.radians(phi_1))*math.tan(math.radians(phi_1)) + math.tan(math.radians(phi_2))*math.tan(math.radians(phi_2)) - 2*(meu*r/rho)*math.tan(math.radians(phi_1))*math.tan(math.radians(phi_2))) / (R*R)
    # si = math.asin(v*iz)*57.29578
    # ai = math.atan((-v*ix)/(-v*iy))*57.29578

    # if ai<min(a1,a2):
    #   ai+=180
    # elif ai>max(a1,a2):
    #   ai-=180
    
    # # Check on Wedge Geometry
    # if p*iz < 0 or n*q*iz < 0 :
    #   print("NO Wedge is Formed ")

    # if(e*ita*q5*iz < 0 or h5<0 or abs((m5*h5)/(m*h))>1 or abs((n*q5*m5*h5)/(n5*q*m*h))>1):
    #   print("Tension Crack Invalid")
    #   print(p*iz ,n*q*iz)

    # A1 = (abs(m*q)*h*h - abs(m5*q5)*h5*h5)/(2*abs(p))
    # A2 = ((abs(q)*m*m*h*h)/abs(n) - (abs(q5)*m5*m5*h5*h5)/abs(n5))/(2*abs(p))
    # A5 = (abs(m5*q5)*h5*h5)/(2*n5)
    # W = gamma_rock*((q*q*m*m*h*h*h/abs(n) - q5*q5*m5*m5*h5*h5*h5/abs(n5))/abs(6*p))
    # print(A1 , A2 ,A5 , W)

    # #Water Pressure
    # # 1. With tension Crack
    # u1 = (gamma_water*h5*abs(m5))/(3*dz)
    # u2 = u1
    # u5 = u1
    # V = u5*A5*ita*(e/abs(e))
    # print(u1 , V)

    # #2. Without Tention Crack
    # u1 = (gamma_water*h5*abs(m5))/(3*dz)
    # u2 = u1
    # u5 = u1
    # V = u5*A5*ita*(e/abs(e))
    # print(u1 , V)

    # #Normal Reactions on Plane 1 and Plane 2 Contact on both planes
    # N1 = rho*(W*kz + T*(r*v - s) + E*(e*ue - se) + (V*(r*v5 - s5))) - u1*A1
    # N2 = meu*(W*lz + T*(r*s - v) + E*(r*se -ue) + V*(r*s5 - v5)) - u2*A2
    # print(N1 , N2)
    # if(N1 < 0 and N2<0):
    #   print("Factor of Safety is Zero ")

    # if(N1 > 0 and N2 < 0):
    #   Na = W*az - T*s - E*se - V*s5 - u1*A1*r
    #   Sx = T*tx + E*ex + Na*ax + V*f5x + u1*A1*bx
    #   Sy = T*ty + E*ey + Na*ay + V*f5y + u1*A1*by
    #   Sz = (T*tz + E*ez + Na*az + V*f5z + u1*A1*bz) + W
    #   Sa = math.sqrt(Sx*Sx + Sy*Sy + Sz*Sz)
    #   Qa = (Na - u1*A1)*math.tan(phi_1) + c1*A1
    #   Factor_of_safety_1 = (Qa / Sa)
    #   print(Factor_of_safety_1)
    # if(N1 < 0 and N2 > 0):
    #   Nb = (W*bz - T*u - E*ue - u2*A2*r)
    #   Sx = (T*tx + E*ex + Nb*bx + V*f5x + u2*A2*ax)
    #   Sy = (T*ty + E*ey + Nb*by + V*f5y + u2*A2*ay)
    #   Sz = (T*tz + E*ez + Nb*bz + V*f5z + u2*A2*az) + W
    #   Sb = math.sqrt(Sx*Sx + Sy*Sy + Sz*Sz)
    #   Qb = (Nb - u2*A2)*math.tan(phi_2) + c2*A2
    #   Factor_of_safety_2 = (Qb/Sb)
    #   print(Factor_of_safety_2)
    # if(N1 > 0 and N2 > 0):
    #   S = v*(W*iz - T*w - E*we - V*w5)
    #   Q = N1*math.tan(math.radians(phi_1)) + N2*math.tan(math.radians(phi_2)) + c1*A1 + c2*A2
    #   Factor_of_safety_3 = (Q/S)
    #   # print(S , Q ,iz , w5 ,v)
     
    # # render the HTML template and pass in the base64 encoded image
    return render_template('app.html')

if __name__ == "__main__":
    app.run(debug=True)

