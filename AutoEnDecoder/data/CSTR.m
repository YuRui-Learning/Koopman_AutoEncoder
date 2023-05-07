cita = 20; k = 300; M = 5; xf = 0.3947; xc = 0.3816; arfa = 0.117; Ts = 0.2;


a = 0.1; b = 2;
U=zeros(10,50);
x1=zeros(11,50);
x2=zeros(11,50);

for i=1:50
    u=a+(b-a)*rand(10,1);
    X=rand(1,2);
    x1(1,i)=X(1,1);
    x2(1,i)=X(1,2);
    for j =1:10
        U(j,i)=u(j,1);
        x1(j+1,i)=x1(j,i)+(Ts/cita)*(1-x1(j,i))-Ts*k*x1(j,i)*exp(-M/x2(j,i));
        x2(j+1,i)=x2(j,i)+(Ts/cita)*(xf-x2(j,i))+Ts*k*x1(j,i)*exp(-M/x2(j,i))-Ts*arfa*U(j,i)*(x2(j,i)-xc); 
    end

end


