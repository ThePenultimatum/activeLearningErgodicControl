clear;


%%%%
global T x0 epsilon resolution dt sigma mu L K rowres colres gammaKs R qRegularization controls ergodicMeasure;
global phik trajectory deriv phix cost Q P1 Amats Bmats vs zs N z0 v0 xdest timevals littlea littleb allHs;
%%%%

i = 0;
T = 10;
%x0 = [0; 1];
x0 = [1; 2];
u1_init = 0.001;
u2_init = 0.001;
initCondsVect = [1; 2; 1; 0.1];


allHs = [];
    
for xk=0:K
    for yk=0:K
        ks = [xk; yk];
        allHs(xk+1,yk+1) = getHK(ks);
    end
end


resolution = 1;
dt = 1/resolution;
N = T/dt;
controls = [];
trajectory = [];
timevals = [];
xdest = [];
zs = [];
vs = [];
z0 = [0; 0];
v0 = [0; 0];
prev = x0;
for i=1:(T/dt)
    controls = [controls; u1_init, u2_init];
    trajectory = [trajectory; prev(1) + dt * u1_init, prev(2) + dt * u2_init];
    prev = [prev(1) + dt * u1_init, prev(2) + dt * u2_init];
    zs = [zs; 0, 0];
    vs = [vs; 0, 0];
    timevals(i) = dt * i;
end
xdest = transpose(trajectory);
epsilon = 0.1;
R = [1 0; 0 1];
Q = [1 0; 0 1];
P1 = [1 0; 0 1];
Amats = [];
Bmats = [];
sigma = [1 0; 0 1];
mu = 1;%[1; 1];%0;
L = 2;%2;
qRegularization = 1;
K = 10;
rowres = 10;
colres = 10;
gammaKs = [];
for ki=0:K
    for kj=0:K
        %newone = getGammaK(k);
        %gammaKs = [gammaKs; newone];
        newone = getGammaKiKj(ki, kj);
        gammaKs(ki+1,kj+1) = newone;
    end
end
phix = phi();
ergodicMeasure = getErgodicMeasure()
cost = costJ();
ergodicMeasure = getErgodicMeasure();
deriv = 10;%derivOfCost() %10; % initialize dj(eta_i) dot zeta_i
alpha = 0.4;
beta = 0.7;

%%%% Main loop

iters = 0;

while ((norm(deriv)) > epsilon) & iters < 1000
    
    
    %%% calculate  A and B matrices
    for i=1:(T/dt)
      At = Amat(trajectory, controls, i);
      Amats(:,:,i) = At;
      Bt = Bmat(trajectory, controls, i);
      Bmats(:,:,i) = Bt;
    end
    littlea = transpose(getLittlea());
    %%% Calculate P
    [TP, P] = ode45(@(t,P)solvepval(t, P, Q, R, Amats, Bmats, trajectory, controls), linspace(T,0,T/dt), P1);
    tmpP = P((T/dt),:);
    
    %%% Calculate r0 and r
    r0 = [tmpP(1,1:2); tmpP(1,3:4)] * z0;%[tmpP(1,1:3); tmpP(1,4:6); tmpP(1, 7:9)] * z0; %P1 * (trajectory(:,N) - xdest(:,N));%[tmpP(1,1:3); tmpP(1,4:6); tmpP(1, 7:9)] * z0;
    
    [Tr, r] = ode45(@(t,r)solverval(t, r, P, R, Q, Amats, Bmats, trajectory, controls), linspace(T,0,T/dt), r0);
    
    %%% Calculate z
    x0valForZdot = [0; 0;];% 0];%[trajectory(1,1); trajectory(2,1); trajectory(3,1)];
    [Tz, z] = ode45(@(t,z)getZdot(t, z, flip(P), R, Q, Amats, Bmats, trajectory, controls, flip(r)), linspace(0,T,T/dt), x0valForZdot);
    zs = z;
    
    %%% Calculate v
    v = getV(R, Amats, Bmats, flip(P), z, flip(r), trajectory, controls, Q);
    vs = v;
    
    %%% armijo
    
    %%% setup armijo initial conditions
    jnew = 9999999;
    jinit = cost;
    beta  = 0.7;
    alpha = 0.4;
    armijocount = 0;
    %%% armijo: while cost of current cols is more than cost of taking step
    %while (_ - _ < _ & armijocount < armijomax)
    %    beta = 0.7 ^ armijocount;
    %    armijocount = armijocount + 1;
    %end
    
    %%% UPDATES to trajectory and controls
    
    
    
    %%%%%%%
    %temporary
    gamma = 0.01;
    oldControls = controls + transpose(v) * gamma;
    oldTraj = [];
    prev = x0;
    for i=1:(T/dt)
        oldTraj = [oldTraj; prev(1) + dt * oldControls(i,1), prev(2) + dt * oldControls(i,2)];
        prev = [prev(1) + dt * u1_init, prev(2) + dt * u2_init];
    end
    %%%%%%%
    
    trajectory = oldTraj;
    controls = oldControls;
    
    %%% Update counter
    iters = iters + 1
    
    %%% Update ergodic measure
    ergodicMeasure = getErgodicMeasure()
    cost = costJ()
    deriv = derivOfCost()
    
    
    
    
    
end


%plot(T_t(:,1), T_t(:,2)); %%%%%%%%%%%%%%%%%%%%%% Plotting like x = rows, y = cols
plot(trajectory(:,1), trajectory(:,2),".");
title("Trajectory for hw5 problem 1");
%xlim([-1 5]); 
%ylim([-1 5]);
xlabel("x");
ylabel("y");
%hold on;
%plot(T_t(1,1), T_t(1,2),'.r','MarkerSize',40);
%plot(doorLocation(1), doorLocation(2),'.b','MarkerSize',40);
%hold off;


%%%% Functions

function lita = getLittlea()
    global controls ergodicMeasure qRegularization times R zs vs trajectory gammaKs K allHs;
    %[t, x] = ode45(@(t,x) [0 1; -1 -b]*x, linspace(0,T,T+1), x0);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %t = times;
    %trajectory = x;
    
    sumSoFar = 0;
    
    for xk=0:K
        for yk=0:K
            ks = [xk; yk];
            %
            gammaIJ = gammaKs(xk+1, yk+1);
            %
            h = allHs(xk+1, yk+1); % getHK(ks);
            %
            fkx = getFkx(trajectory, ks, h);
            %
            ck = getCks(fkx);
            %
            phik = getPhik(ks, h);
            %
            %phik = getPhik(fkx);
            %
            termWithoutZ = getDFkxZWithoutZ(fkx, zs);
            sumSoFar = sumSoFar + gammaIJ * termWithoutZ;
        end
    end
    de = 2*sumSoFar;
    lita = de;
end

function zterm = getDFkxZWithoutZ(fkx, z)
    global T dt;
    %fun = @(t) fkx(t);
    %ck = integral(fun, 1, T/dt+1); %0;%(1/T) *
    %ck = (1/T) * [sum(fkx(1,:)) ; sum(fkx(2,:))];
    zterm = (1/T) * (fkx);%(1,:));
end

function zterm = getDFkxZ(fkx, z)
    global T dt;
    %fun = @(t) fkx(t);
    %ck = integral(fun, 1, T/dt+1); %0;%(1/T) *
    %ck = (1/T) * [sum(fkx(1,:)) ; sum(fkx(2,:))];
    zterm = (1/T) * sum(fkx * transpose(z));%(1,:));
end

function dfkx = getDFkx(x, k, h)
    global L;
    res = [];
    normalizer = 1/h;
    resi = normalizer * 1;
    
    x1partDx1 = -sin(k(1) * pi * x(1,:) / L);
    x2partDx1 = cos(k(2) * pi * x(2,:) / L);
    x1partDx2 = cos(k(1) * pi * x(1,:) / L);
    x2partDx2 = -sin(k(2) * pi * x(2,:) / L);
    tot = resi * pi * (1/L) * (k(1) * (x1partDx1 .* x2partDx1) + k(2) * (x1partDx2 .* x2partDx2));
    fkx = tot;
end

function de = derivOfErg()
    global controls ergodicMeasure qRegularization times R zs vs trajectory gammaKs K allHs;
    %[t, x] = ode45(@(t,x) [0 1; -1 -b]*x, linspace(0,T,T+1), x0);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %t = times;
    %trajectory = x;
    
    sumSoFar = 0;
    
    for xk=0:K
        for yk=0:K
            ks = [xk; yk];
            %
            gammaIJ = gammaKs(xk+1, yk+1);
            %
            h = allHs(xk+1, yk+1); % getHK(ks);
            %
            fkx = getFkx(trajectory, ks, h);
            %
            ck = getCks(fkx);
            %
            phik = getPhik(ks, h);
            %
            %phik = getPhik(fkx);
            %
            termWithZ = getDFkxZ(fkx, zs);
            sumSoFar = sumSoFar + gammaIJ * (ck - phik) * termWithZ;
        end
    end
    de = 2*sumSoFar;
end

function d = derivOfCost()
    global controls ergodicMeasure qRegularization timevals R zs vs;
    sumsofar = 0;
    for i=1:(length(timevals))
        sumsofar = sumsofar + controls(i,:) * R * vs(:,i);
    end
    regularizationTerm = 0.5 * sumsofar;
    dErg = derivOfErg();
    dj = qRegularization * dErg + regularizationTerm;
    d = dj;
end

function j = costJ()
    global controls ergodicMeasure qRegularization times R;
    sumsofar = 0;
    for i=1:(length(times))
        sumsofar = sumsofar + transpose(controls(i,:)) * R * controls(i,:);
    end
    regularizationTerm = 0.5 * sumsofar;
    j = qRegularization * ergodicMeasure + regularizationTerm;
end

function ergodicmeasureval = getErgodicMeasure()
    global trajectory gammaKs K allHs;
    %[t, x] = ode45(@(t,x) [0 1; -1 -b]*x, linspace(0,T,T+1), x0);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %t = times;
    %trajectory = x;
    
    sumSoFar = 0;
    
    for xk=0:K
        for yk=0:K
            ks = [xk; yk];
            %
            gammaIJ = gammaKs(xk+1, yk+1);
            %
            h = allHs(xk+1, yk+1); % getHK(ks);
            %
            fkx = getFkx(trajectory, ks, h);
            %
            ck = getCks(fkx);
            %
            phik = getPhik(ks, h);
            %
            %phik = getPhik(fkx);
            sumSoFar = sumSoFar + gammaIJ * (norm(ck - phik))^2;
        end
    end
    
    ergodicmeasureval = sumSoFar;
end

function pkx = getPhik(ks, hk)%(fkx)
    global phix rowres colres L;
    %phix;
    %fkx;
    %pkx = sum(phix) * transpose(fkx);
    
    sumsofar = 0;
    drow = L / rowres;
    dcol = L / colres;
    for a=0:(rowres+1)
        for b=0:(colres+1)
            drowab = drow * a;
            dcolab = dcol * b;
            elem = (1/hk)*cos(ks(1)*pi*drowab/L) * cos(ks(2)*pi*dcolab/L);
            sumsofar = sumsofar + elem;
        end
    end
    
    pkx = phix(ks(1)+1, ks(2)+1)*sumsofar;%/(colres*rowres);
end

function ck = getCks(fkx)
    global T dt;
    fkx;
    %fun = @(t) fkx(t);
    %ck = integral(fun, 1, T/dt+1); %0;%(1/T) *
    %ck = (1/T) * [sum(fkx(1,:)) ; sum(fkx(2,:))];
    ck = (1/T) * sum(fkx);%(1,:));
end

function fkx = getFkx(x, k, h)
    global L
    1;
    %h = allHs(ks(1),ks(2)); % getHK(k);
    %L = 1;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%????????????????????????????/
    res = [];
    normalizer = 1/h;
    resi = normalizer * 1;
    
    x1part = cos(k(1) * pi * x(1,:) / L);
    x2part = cos(k(2) * pi * x(2,:) / L);
    tot = resi * (x1part .* x2part);
    fkx = tot;
end

function hk = getHK(ks)
    global L 
    fun = @(x1,x2) (cos(ks(1) * pi * x1 / L).^2).*((cos(ks(2) * pi * x2 / L)).^2);
    hk = sqrt(integral2(fun, 0, L, 0, L));
end

function g = getGammaKiKj(i, j) 
    %%%%%%%%%%% this assumes that k is a column vector input or a scalar
    usek = [i; j];
    n = 2;
    g = (1+norm(usek)^2)^(-1*((n+1)/2));
end

function phi_x = phi()
  global sigma mu L K;
  newk = K + 1;
  xval = [linspace(0, L, newk); linspace(0, L, newk)];
  phi_x = (1/(sqrt(det(2*pi*sigma))))*exp(-0.5*transpose(xval-mu)*inv(sigma)*(xval-mu));
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%

function norm = normOfRowVect(vectVal)
  % this is of the form [[x1,.....,xn]] * transpose([[x1,....,xn]]);
  norm = vectVal * transpose(vectVal);
end

function norm = matrixNorm(matrix, nrows)
  % norm of natrix with n rows
  sumval = 0;
  for i = 1:nrows
      sumval = sumval + normOfRowVect(matrix(i,:));
  end
  norm = sumval;
end

%%%

function pdot = solvepval(t, P, Q, R, As, Bs, xs, us)
  global N T;
  index = round((t/T)*(N-1)+1);
  P;
  A = As(:,:,index);
  B = Bs(:,:,index);
  
  P = reshape(P, size(A));
  pdot = -1*transpose(A)*P - P*A + P*B*(inv(R))*(transpose(B))*P - Q;
  pdot = pdot(:);
end

function rdot = solverval(t, r, P, R, Q, As, Bs, xs, us)
  global xdest N T littlea;
  index = round((t/T)*(N-1)+1);
  
  xs = transpose(xs);
  us = transpose(us);
  
  newx1 = xs(1,index);
  newx2 = xs(2,index);
  %newx3 = xs(3,index);
  newu1 = us(1,index);
  newu2 = us(2,index);
  
  xdestcols1 = xdest(1,index);
  xdestcols2 = xdest(2,index);
  %xdestcols3 = xdest(3,index);
  xdestcolsToUse = [xdestcols1; xdestcols2];%; xdestcols3];
  
  A = As(:,:,index);
  B = Bs(:,:,index);
  
  littlea;% = transpose(getLittlea());%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%transpose(transpose([newx1; newx2]-xdestcolsToUse) * Q);
  littleb = transpose(transpose([newu1; newu2]) * R);
  Pval = P(index,:);
  newP = [Pval(1:2); Pval(3:4)];%[Pval(1:3); Pval(4:6); Pval(7:9)];
  
  rdot = -1*transpose(A - B * inv(R) * transpose(B) * newP)*r - littlea + newP * B * (inv(R)) * littleb;
  rdot = rdot(:);
  
end

function zdot = getZdot(t, z, P, R, Q, As, Bs, xs, us, rs, littleA)
  global xdest N T trajectory littlea;
  index = round((t/T)*(N-1)+1);
  xs = transpose(xs);
  us = transpose(us);
  A = As(:,:,index);
  B = Bs(:,:,index);
  newx1 = xs(1,index);
  newx2 = xs(2,index);
  %newx3 = xs(3,index);
  newu1 = us(1,index);
  newu2 = us(2,index);
  newr1 = rs(index,1);
  newr2 = rs(index,2);
  %newr3 = rs(index,3); 
  xdestcols1 = xdest(1,index);
  xdestcols2 = xdest(2,index);
  %xdestcols3 = xdest(3,index);
  xdestcolsToUse = [xdestcols1; xdestcols2];% xdestcols3];
  littlea;% = getLittlea();%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%transpose(transpose([newx1; newx2] - xdestcolsToUse) * Q);%; newx3]-xdestcolsToUse) * Q);
  littleb = transpose(transpose([newu1; newu2]) * R);
  Pval = P(index,:);
  newP = [Pval(1:2); Pval(3:4)];%[Pval(1:3); Pval(4:6); Pval(7:9)];
  rval = [newr1, newr2];%[newr1; newr2; newr3];
  
  zdot = A * z + B * (-1 * inv(R) * transpose(B) * newP * z - inv(R) * transpose(B) * transpose(rval) - inv(R) * littleb);
end

function v = getV(R, As, Bs, P, zs, rs, xs, us, Q)
  global T N xdest littlea;
  vs = [];
  xs = transpose(xs);
  us = transpose(us);
  for i=1:length(rs)
    A = As(:,:,i);
    B = Bs(:,:,i);
    Pval = P(i,:);
    %newP = [Pval(1:3); Pval(4:6); Pval(7:9)];
    newP = [Pval(1:2); Pval(3:4)];
    newx1 = xs(1,i);
    newx2 = xs(2,i);
    %newx3 = xs(3,i);
    newu1 = us(1,i);
    newu2 = us(2,i);
    xdestcols1 = xdest(1,i);
    xdestcols2 = xdest(2,i);
    %xdestcols3 = xdest(3,i);
    xdestcolsToUse = [xdestcols1; xdestcols2];% xdestcols3];
    littlea;% = getLittlea();%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%transpose(transpose([newx1; newx2]-xdestcolsToUse) * Q);% newx3]-xdestcolsToUse) * Q);
    littleb = transpose(transpose([newu1; newu2]) * R);
    z = transpose(zs(i,:));
    rval = transpose(rs(i,:));
    vs(:,i) = -1 * inv(R) * transpose(B) * newP * z - inv(R) * transpose(B) * rval - inv(R) * littleb;
  end
  v = vs;%-1 * inv(R) * transpose(B) * newP * z - inv(R) * transpose(B) * rval - inv(R) * b;
end

%%%

function Amatv = Amat(x, u, index)
  % D_1(f(x,u))
  Amatv = [0 0; 0 0];%[0 0 -sin(x(3,index))*u(1,index); 0 0 cos(x(3,index))*u(1,index); 0 0 0];
end

function Bmatv = Bmat(x, u, index)
  % D_2(f(x,u))
  Bmatv = [1 0; 0 1];%[cos(x(3,index)) 0; sin(x(3,index)) 0; 0 1];
end

function xvectdot = Fvectdot(x, u, index)
  xvectdot = [cos(x(3, index)) * u(1, index); sin(x(3, index)) * u(1, index); u(2, index)];
end

function xvectdot = Fsinglevectdot(x, u)
  xvectdot = [cos(x(3)) * u(1); sin(x(3)) * u(1); u(2)];
end

function xdest = Fdest(i)
  global dtval;

  xdest = [dtval * 2 * i / pi; 0; pi/2];
end