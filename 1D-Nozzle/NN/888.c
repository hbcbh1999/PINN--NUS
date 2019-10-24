#include<stdio.h>
#include<stdlib.h>
#include<math.h>


void main()
{

FILE *datafile; 
FILE *rssfile;
int i,j,k,q;
float U[110][3],Fp[110][3],Fn[110][3],F[110][3],M[110],L1[110],L2[110],L3[110],wp[110],wn[110],z[103];
float p_abs[103],u_abs[103],rho[110],rho_abs[110],p[110],u[110],a[110],A[110],S[110],rold[103],rn[103],rss1[110],rss2=0,rss3,pmax=0,rhomax=0,umax=0,Mmax=0;
float u_plus_a=0,dt,dx=0.01,g,R,al,pl,ul,rhol,Ml,ptl,pratio,R2,R3,R1;

float frac=0.0;

char filename[100]="cdnozzle.txt"; 
char filename1[100] ="residue.txt";

datafile=fopen(filename,"w");

for (frac=0.3; frac<=0.9; frac += 0.03){

rssfile=fopen(filename1,"w");

g=1.4,R=287.16;   

rhol=1,ul=0.1,pl=1,al=sqrt(g*1/rhol),Ml=ul/al,ptl=1.005; // values are to be changed

R2=ul+(5*al);

//Intial Condition:
for(i=1;i<=100;i++)
{

A[i]=1+2.2*pow(((3*dx*i)-1.5),2);

rho[i]=1;
u[i]=0.1;
p[i]=1;
a[i]=sqrt(g*p[i]/rho[i]);
M[i]=u[i]/a[i];

rold[i]=rho[i];
U[i][0]=A[i]*rho[i];
U[i][1]=A[i]*rho[i]*u[i];
U[i][2]=A[i]*((p[i]/(g-1))+(0.5*rho[i]*u[i]*u[i]));

L1[i]=u[i];
L2[i]=u[i]+a[i];
L3[i]=u[i]-a[i];
wp[i]=(3-g)*a[i]*a[i]*0.5*L2[i]/(g-1);   
wn[i]=(3-g)*a[i]*a[i]*0.5*L3[i]/(g-1);


	Fp[i][0]=A[i]*(rho[i]*(g-1)*(L1[i]/g)+(rho[i]*L2[i]*0.5/g));                                                
        Fp[i][1]=A[i]*(rho[i]*L1[i]*L1[i]*((g-1)/g)+(rho[i]*L2[i]*L2[i]*0.5/g));                                    
        Fp[i][2]=A[i]*(rho[i]*L1[i]*L1[i]*L1[i]*((g-1)/g)+(rho[i]*L2[i]*L2[i]*L2[i]*0.25/g)+(rho[i]*0.5*wp[i]/g));   
	Fn[i][0]=A[i]*(rho[i]*L3[i]*.5/g);
        Fn[i][1]=A[i]*(rho[i]*L3[i]*L3[i]*.5/g);
        Fn[i][2]=A[i]*(rho[i]*L3[i]*L3[i]*L3[i]*(0.25/g)+(rho[i]*0.5*wn[i]/g));
        F[i][0]=A[i]*rho[i]*u[i];
	F[i][1]=A[i]*(rho[i]*u[i]*u[i]+p[i]);
	F[i][2]=A[i]*((p[i]*(g/(g-1))+0.5*rho[i]*u[i]*u[i])*u[i]);

	dt=0.9*0.01/(u[1]+a[1]);
	
	

      // printf("%d\t%f\t%f\t%f\t%f\t%f\n",i,dt,dx,a[i],S[i],M[i]); 

       //printf("%d\t%f\t%f\t%f\n",i,U[i][0],U[i][1],U[i][2]);
       //printf("%d\t%f\n",i,dt);
}
       

rss3=1;

//calculation:
for(j=1;rss3>=0.00001;j++)
 {

rss2=0;

A[0]=A[1];
A[101]=A[100];

R3=u[1]-(2*a[1]/(g-1));

u[0]=0.5*(R2+R3);
a[0]=0.1*(R2-R3);
M[0]=u[0]/a[0];
pratio=pow((1+(0.2*M[0]*M[0])),3.5);
p[0]=ptl/pratio;
rho[0]=g*p[0]/(a[0]*a[0]);

p[101]=frac*ptl;
rho[101]=rho[100]*pow((p[101]/p[100]),0.714);
a[101]=sqrt(g*p[101]/rho[101]);
u[101]=u[100]+5*(a[100]-a[101]);


//printf("%d\t%f\t%f\t%f\t%f\t%f\n",j,R2,R3,rho[101],a[0],M[0]);

L1[0]=u[0];
L2[0]=u[0]+a[0];
L3[0]=u[0]-a[0];
wp[0]=(3-g)*a[0]*a[0]*0.5*L2[0]/(g-1);   
wn[0]=(3-g)*a[0]*a[0]*0.5*L3[0]/(g-1);


	Fp[0][0]=A[0]*(rho[0]*(g-1)*(L1[0]/g)+(rho[0]*L2[0]*0.5/g));                                                
        Fp[0][1]=A[0]*(rho[0]*L1[0]*L1[0]*((g-1)/g)+(rho[0]*L2[0]*L2[0]*0.5/g));                                    
        Fp[0][2]=A[0]*(rho[0]*L1[0]*L1[0]*L1[0]*((g-1)/g)+(rho[0]*L2[0]*L2[0]*L2[0]*0.25/g)+(rho[0]*0.5*wp[0]/g));   
	Fn[0][0]=A[0]*(rho[0]*L3[0]*.5/g);
        Fn[0][1]=A[0]*(rho[0]*L3[0]*L3[0]*.5/g);
        Fn[0][2]=A[0]*(rho[0]*L3[0]*L3[0]*L3[0]*(0.25/g)+(rho[0]*0.5*wn[0]/g));
        F[0][0]=A[0]*rho[0]*u[0];
	F[0][1]=A[0]*(rho[0]*u[0]*u[0]+p[0]);
	F[0][2]=A[0]*((p[0]*(g/(g-1))+0.5*rho[0]*u[0]*u[0])*u[0]);


L1[101]=u[101];
L2[101]=u[101]+a[101];
L3[101]=u[101]-a[101];
wp[101]=(3-g)*a[101]*a[101]*0.5*L2[101]/(g-1);   
wn[101]=(3-g)*a[101]*a[101]*0.5*L3[101]/(g-1);


	Fp[101][0]=A[101]*(rho[101]*(g-1)*(L1[101]/g)+(rho[101]*L2[101]*0.5/g));                                                
        Fp[101][1]=A[101]*(rho[101]*L1[101]*L1[101]*((g-1)/g)+(rho[101]*L2[101]*L2[101]*0.5/g));                                    
        Fp[101][2]=A[101]*(rho[101]*L1[101]*L1[101]*L1[101]*((g-1)/g)+(rho[101]*L2[101]*L2[101]*L2[101]*0.25/g)+(rho[101]*0.5*wp[101]/g));   
	Fn[101][0]=A[101]*(rho[101]*L3[101]*.5/g);
        Fn[101][1]=A[101]*(rho[101]*L3[101]*L3[101]*.5/g);
        Fn[101][2]=A[101]*(rho[101]*L3[101]*L3[101]*L3[101]*(0.25/g)+(rho[101]*0.5*wn[101]/g));
        F[101][0]=A[101]*rho[101]*u[101];
	F[101][1]=A[101]*(rho[101]*u[101]*u[101]+p[101]);
	F[101][2]=A[101]*((p[101]*(g/(g-1))+0.5*rho[101]*u[101]*u[101])*u[101]);



	//printf("%d\t%f\t%f\t%f\t%f\t%f\n",i,u[101],p[101],rho[101],a[101],T[101]);
    


for(i=1;i<=100;i++)
  {
   	if(M[i]>=-1 && M[i]<=1) // Mach no condition
  	{ 
	  	 
  	U[i][0]=U[i][0]-((dt/dx)*(Fp[i][0]-Fp[i-1][0]))-((dt/dx)*(Fn[i+1][0]-Fn[i][0]));
  	U[i][1]=U[i][1]-((dt/dx)*(Fp[i][1]-Fp[i-1][1]))-((dt/dx)*(Fn[i+1][1]-Fn[i][1]))+(dt/dx)*p[i]*0.5*(A[i+1]-A[i-1]);
  	U[i][2]=U[i][2]-((dt/dx)*(Fp[i][2]-Fp[i-1][2]))-((dt/dx)*(Fn[i+1][2]-Fn[i][2]));
	//printf("%d\t%f\t%f\t%f\n",i,U[i][0],U[i][1],U[i][2]);
 	}

        else if(M[i]<-1) 
        {
	U[i][0]=U[i][0]-((dt/dx)*(F[i+1][0]-F[i][0]));
  	U[i][1]=U[i][1]-((dt/dx)*(F[i+1][1]-F[i][1]))+(dt/dx)*p[i]*0.5*(A[i+1]-A[i-1]);
  	U[i][2]=U[i][2]-((dt/dx)*(F[i+1][2]-F[i][2]));
	}

   
	 else if(M[i]>1) 
        {
	U[i][0]=U[i][0]-((dt/dx)*(F[i][0]-F[i-1][0]));
  	U[i][1]=U[i][1]-((dt/dx)*(F[i][1]-F[i-1][1]))+(dt/dx)*p[i]*0.5*(A[i+1]-A[i-1]);
  	U[i][2]=U[i][2]-((dt/dx)*(F[i][2]-F[i-1][2]));
	}

	

    	
}// i loop												
        
  	// updating...

	for(i=1;i<=100;i++)
        {

	U[i][0]=U[i][0]/A[i];
	U[i][1]=U[i][1]/A[i];
	U[i][2]=U[i][2]/A[i];

//printf("%d\t%f\t%f\t%f\n",i,U[i][0],U[i][1],U[i][2]);

	rho[i]=U[i][0];
	
  	u[i]=U[i][1]/U[i][0];
	p[i]=(U[i][2]-(U[i][1]*U[i][1]*0.5/U[i][0]))*(g-1);
 	p_abs[i]=fabs(p[i]);
	rho_abs[i]=fabs(rho[i]);
	u_abs[i]=fabs(u[i]);

	
	rn[i]=rho[i];

    	a[i]=sqrt(g*(p_abs[i])/(rho_abs[i]));
 	L1[i]=u[i];
  	L2[i]=u[i]+a[i];
  	L3[i]=u[i]-a[i];
  	wp[i]=(3-g)*a[i]*a[i]*0.5*L2[i]/(g-1);
  	wn[i]=(3-g)*a[i]*a[i]*0.5*L3[i]/(g-1);
  	M[i]=u[i]/a[i];  
    	//printf("%d\t%f\t%f\t%f\t%f\t%f\n",i,dt,M[i],rho[i],u[i],p[i]);
  	Fp[i][0]=A[i]*(rho[i]*(g-1)*(L1[i]/g)+(rho[i]*L2[i]*0.5/g));
  	Fp[i][1]=A[i]*(rho[i]*L1[i]*L1[i]*((g-1)/g)+(rho[i]*L2[i]*L2[i]*0.5/g));
  	Fp[i][2]=A[i]*(rho[i]*L1[i]*L1[i]*L1[i]*((g-1)/g)+(rho[i]*L2[i]*L2[i]*L2[i]*0.25/g)+(rho[i]*0.5*wp[i]/g));
 	Fn[i][0]=A[i]*(rho[i]*L3[i]*0.5/g);
  	Fn[i][1]=A[i]*(rho[i]*L3[i]*L3[i]*0.5/g);
  	Fn[i][2]=A[i]*(rho[i]*L3[i]*L3[i]*L3[i]*(0.25/g)+(rho[i]*0.5*wn[i]/g));

	F[i][0]=A[i]*rho[i]*u[i];
	F[i][1]=A[i]*(rho[i]*u[i]*u[i]+p[i]);
	F[i][2]=A[i]*((p[i]*(g/(g-1))+0.5*rho[i]*u[i]*u[i])*u[i]);
        
	
        z[i]=u[i]+a[i];

       	U[i][0]=U[i][0]*A[i];
	U[i][1]=U[i][1]*A[i];
	U[i][2]=U[i][2]*A[i];

          	//printf("%d\t%f\t%f\t%f\n",i,U[i][0],U[i][1],U[i][2]);


                // finding dt global...
       		if(u_plus_a<(z[i]))
	  	 {
         	  u_plus_a=z[i];
         	  }
		dt=0.9*0.01/u_plus_a;


	// Residue calculation

      rss1[i]=((rn[i]-rold[i])/rold[i])*((rn[i]-rold[i])/rold[i]);

      rss2=rss2+rss1[i];
	
	rold[i]=rn[i];
      
     
	}// update loop


       //printf("%d\t%f\n",j,rss3);
                	
        rss3=sqrt(rss2/100);

        fprintf(rssfile,"%d\t%f\n",j,rss3);
                 
          
}// j loop

fclose(rssfile);
     

 //opening a file named "cdnozzle" on home folder:

  
  
//   printf("i\tp[i]\t\trho[i]\t\tu[i]\t\tM[i]\n");

for (i=1;i<=100;i++)
  {
//    printf("%d\t%f\t%f\t%f\t%f\n",i,p[i],rho[i],u[i],M[i]);
    fprintf(datafile,"%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",i/1.0, frac, p[i], U[i][0], U[i][1], U[i][2], rho[i], u[i], U[i][2]/U[i][0]);
  } 
  
};

fclose(datafile);

//FILE *plot = popen("gnuplot -persist","w");
//fprintf(plot, "plot 'cdnozzle.txt' using 1:4 title 'rho_S' with lines \n");
//close(plot);
/*
FILE *plot1 = popen("gnuplot -persist","w");
//fprintf(plot1, "plot 'residue.txt' using 1:2 title 'residue' with lines\n");
//close(plot1);
*/

}   // End...

