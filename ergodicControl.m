clear;


%%%% Globals
global T x0 epsilon resolution dt sigma mu L K rowres colres gammaKs R qRegularization controls ergodicMeasure;
global trajectory phix cost Q P1 Amats Bmats vs zs N z0 v0 xdest timevals littlea littleb allHs invR invSigma;
%%%%

%%%% Inits for all parts
T = 10;
x1_init = 0.1;
x2_init = 0.1;
x0 = [x1_init; x2_init];
u1_init = 0.1;
u2_init = 0.01;
initCondsVect = [x1_init; x1_init; u1_init; u2_init];
%%%%

%%%% Timing inits
resolution = 1;
dt = 1/resolution;
N = T/dt;
%%%%

%%%% Inits for ERGODICITY calculations
K = 10;
allHs = hsInit();
L = 4;
qRegularization = 1;
rowres = 10;
colres = 10;
gammaKs = [];
for ki=0:K
    for kj=0:K
        newone = getGammaKiKj(ki, kj);
        gammaKs(ki+1,kj+1) = newone;
    end
end
%%% ERGODICITY prior normal distribution inits
sigma = [0.15 0; 0 0.15];
invSigma = inv(sigma);
mu = [0.2, 0.17];%[1; 1];%0;
phix = phi();
%%%
ergodicMeasure = getErgodicMeasure();
%%%%

%%%% Inits for ILQR
R = [0.01 0; 0 0.01];
invR = inv(R);
Q = [1 0; 0 1];
P1 = [1 0; 0 1];
Amats = [];
Bmats = [];
epsilon = 0.1;
%%%%

%%%% Controls, trajectory, desired trajectory, and perturbation inits
controls = [];
trajectory = [];
timevals = [];
zs = [];
vs = [];
z0 = [0; 0];
v0 = [0; 0];
prev = x0;
controls = [controls; u1_init, u2_init];
trajectory = [trajectory; prev(1), prev(2)];
xdest = transpose(trajectory);
zs = [zs; 0, 0];
vs = [vs; 0, 0];
timevals(1) = 0;
for i=1:(T/dt)
    controls = [controls; u1_init, u2_init];
    trajectory = [trajectory; prev(1) + dt * u1_init, prev(2) + dt * u2_init];
    prev = [prev(1) + dt * u1_init, prev(2) + dt * u2_init];
    zs = [zs; 0, 0];
    vs = [vs; 0, 0];
    timevals(i+1) = dt * i;
end
%%%%

%%%% ILQR cost and derivative inits
cost = costJ();
normControlPerturbations = 9999999;
maxILQRIters = 1000;
%%% Armijo inits for ILQR
jnew = 9999999;
jinit = cost;
beta  = 0.7;
alpha = 0.4;
armijocount = 0;
gamma = 0.01; %%%% default small step instead of Armijo
%%%
%%%%

%%%% Main loop

iters = 0;
ergodicities = [];
costs = [];

while (normControlPerturbations > epsilon) & iters < maxILQRIters
    
    %%%% Calculate descent direction
    %%% calculate  A and B matrices
    for i=1:(T/dt)
      At = Amat(trajectory, controls, i);
      Amats(:,:,i) = At;
      Bt = Bmat(trajectory, controls, i);
      Bmats(:,:,i) = Bt;
    end
    %%% calculate a(t) and b(t)
    littlea = transpose(getLittlea());
    littleb = transpose(controls * R);
    %%% Calculate P
    [TP, P] = ode45(@(t,P)solvepval(t, P, Q, R, Amats, Bmats, flip(trajectory), flip(controls)), linspace(T,0,T/dt), P1);
    tmpP = P((T/dt),:);
    %%% Calculate r0 and r
    r0 = [tmpP(1,1:2); tmpP(1,3:4)] * z0;
    [Tr, r] = ode45(@(t,r)solverval(t, r, P, R, Q, Amats, Bmats, flip(trajectory), flip(controls)), linspace(T,0,T/dt), r0);
    
    %%% Calculate z
    x0valForZdot = [0; 0];
    [Tz, z] = ode45(@(t,z)getZdot(t, z, flip(P), R, Q, Amats, Bmats, trajectory, controls, flip(r)), linspace(0,T,T/dt), x0valForZdot);
    zs = [z; 0, 0];
    
    %%% Calculate v
    v = getV(R, Amats, Bmats, flip(P), zs, flip(r), trajectory, controls, Q);
    vs = [v; 0, 0];
    
    %%% armijo
    
    %%% setup armijo initial conditions
    %jnew = 9999999;
    %jinit = cost;
    %beta  = 0.7;
    %alpha = 0.4;
    %armijocount = 0;
    %%% armijo: while cost of current cols is more than cost of taking step
    %while (_ - _ < _ & armijocount < armijomax)
    %    beta = 0.7 ^ armijocount;
    %    armijocount = armijocount + 1;
    %end
    
    %%% UPDATES to trajectory and controls
    
    
    
    %%%%%%%
    %temporary
    gamma = 0.001;
    oldControls = controls + vs * gamma;
    oldTraj = [];
    prev = x0;
    oldTraj = [oldTraj; prev(1), prev(2)];
    for i=1:(T/dt)
        newx1 = prev(1) + dt * oldControls(i,1);
        newx2 = prev(2) + dt * oldControls(i,2);
        oldTraj = [oldTraj; newx1, newx2];
        prev = [newx1, newx2];
    end
    %%%%%%%
    
    trajectory = oldTraj;
    controls = oldControls;
    
    %%% Update counter
    iters = iters + 1
    
    %%% Update ergodic measure
    ergodicMeasure = getErgodicMeasure();
    ergodicities = [ergodicities, ergodicMeasure];
    cost = costJ()
    costs = [costs, cost];
    %deriv = derivOfCost();
    normControlPerturbations = norm([transpose(vs); transpose(zs)])
    
end

plot(trajectory(:,1), trajectory(:,2),".");
title("Trajectory for hw5 problem 1");
xlabel("x");
ylabel("y");

%%%% Functions

function lita = getLittlea()
    global controls ergodicMeasure qRegularization times R zs vs trajectory gammaKs K allHs T;
    
    sumSoFar = 0;
    
    for xk=0:K
        for yk=0:K
            ks = [xk; yk];
            %
            gammaIJ = gammaKs(xk+1, yk+1);
            %
            h = allHs(xk+1, yk+1); % getHK(ks)
            %
            fkx = getFkx(trajectory, ks, h);
            %
            ck = getCks(fkx);
            %
            phik = getPhik(ks, h);
            %
            termWithZ = ck - phik;
            %
            Dfkx = getDFkx(trajectory, ks, h);
            %
            termWithoutZ = getDFkxZWithoutZ(Dfkx);
            
            sumSoFar = sumSoFar + gammaIJ * termWithZ * termWithoutZ;
        end
    end
    de = (2/T)*sumSoFar;
    lita = de;
end

function zterm = getDFkxZWithoutZ(fkx)
    global T;
    zterm = (1/T) * (fkx);%(1,:));
end

function zterm = getDFkxZ(fkx, z)
    global T dt;
    zterm = (1/T) * sum(fkx * transpose(z));%(1,:));
end

function dfkx = getDFkx(x, k, h)
    global L;
    res = [];
    normalizer = 1/h;
    resi = normalizer * 1;
    x1partDx1 = -sin(k(1) * pi * x(:,1) / L);
    x2partDx1 = cos(k(2) * pi * x(:,2) / L);
    x1partDx2 = cos(k(1) * pi * x(:,1) / L);
    x2partDx2 = -sin(k(2) * pi * x(:,2) / L);
    %tot = resi * pi * (1/L) * (k(1) * (x1partDx1 .* x2partDx1) + k(2) * (x1partDx2 .* x2partDx2));
    tot = resi * pi * (1/L) * (k(1) * [transpose(x1partDx1) * x2partDx1; transpose(x1partDx2) * x2partDx2]);
    tot;
    dfkx = tot;
end

function de = derivOfErg()
    global controls ergodicMeasure qRegularization times R zs vs trajectory gammaKs K allHs;
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
        sumsofar = sumsofar + controls(i,:) * R * transpose(vs(i,:));
    end
    regularizationTerm = 0.5 * sumsofar;
    dErg = derivOfErg();
    dj = qRegularization * dErg + regularizationTerm;
    d = dj;
end

function j = costJ()
    global controls ergodicMeasure qRegularization times R vs zs Q;
    sumsofar = 0;
    for i=1:(length(times))
        sumsofar = sumsofar + transpose(controls(i,:)) * R * controls(i,:);
    end
    regularizationTerm = 0.5 * sumsofar;
    j = qRegularization * ergodicMeasure + regularizationTerm + 0.5*(sum(sum(transpose(zs*Q)*zs)) + sum(sum(transpose(vs*R)*vs)));
end

function ergodicmeasureval = getErgodicMeasure()
    global trajectory gammaKs K allHs;    
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

function pkx = getPhik(ks, hk)
    global phix rowres colres L K;
    sumsofar = 0;
    drow = L / K;%rowres;
    dcol = L / K;%colres;
    for a=0:(K-1)%(rowres+1)
        for b=0:(K-1)%(colres+1)
            drowab = drow * a;
            dcolab = dcol * b;
            elem = (1/hk)*cos(ks(1)*pi*drowab/L) * cos(ks(2)*pi*dcolab/L);
            %sumsofar = sumsofar + elem;
            sumsofar = sumsofar + phix(a+1, b+1) * elem;
        end
    end
    
    pkx = sumsofar;%phix(ks(1)+1, ks(2)+1) * sumsofar;
end

function ck = getCks(fkx)
    global T dt;
    fkx;
    ck = (1/T) * sum(fkx);%(1,:));
end

function fkx = getFkx(x, k, h)
    global L
    normalizer = 1/h;
    resi = normalizer * 1;
    x1part = cos(k(1) * pi * x(:,1) / L);
    x2part = cos(k(2) * pi * x(:,2) / L);
    tot = resi * (x1part .* x2part);
    fkx = tot;
end

function hk = getHK(ks)
    global L 
    fun = @(x1,x2) (cos(ks(1) * pi * x1 / L).^2).*((cos(ks(2) * pi * x2 / L)).^2);
    hk = sqrt(integral2(fun, 0, L, 0, L));
end

function hs = hsInit()
    global K;
    allHs = [];
    for xk=0:K
        for yk=0:K
            ks = [xk; yk];
            allHs(xk+1,yk+1) = getHK(ks);
        end
    end
    hs = allHs;
end

function g = getGammaKiKj(i, j) 
    %%%%%%%%%%% this assumes that k is a column vector input or a scalar
    usek = [i; j];
    n = 2;
    g = (1+norm(usek)^2)^(-1*((n+1)/2));
end

function phi_x = phi()
  global sigma mu L K invSigma;
  newk = K + 1;
  xval = [linspace(0, L, newk); linspace(0, L, newk)];
  phimat = zeros(newk, newk);
  for i=1:newk
      for j=1:newk
          xval = [i*L/newk; j*L/newk];
          phimat(i, j) = ((det(2*pi*sigma))^(-0.5))*exp(-0.5*transpose(xval-transpose(mu))*invSigma*(xval-transpose(mu)));
      end
  end
  phi_x = phimat;%((det(2*pi*sigma))^(-0.5))*exp(-0.5*transpose(xval-transpose(mu))*inv(sigma)*(xval-transpose(mu)));
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function pdot = solvepval(t, P, Q, R, As, Bs, xs, us)
  global N T invR;
  index = round((t/T)*(N-1)+1);
  P;
  A = As(:,:,index);
  B = Bs(:,:,index);
  
  P = reshape(P, size(A));
  pdot = transpose(A)*P - P*A + P*B*(invR)*(transpose(B))*P - Q;
  pdot = pdot(:);
end

function rdot = solverval(t, r, P, R, Q, As, Bs, xs, us)
  global N T littlea littleb invR;
  index = round((t/T)*(N-1)+1);
  
  us = transpose(us);
  
  A = As(:,:,index);
  B = Bs(:,:,index);

  Pval = P(index,:);
  newP = [Pval(1:2); Pval(3:4)];
  
  rdot = -1*transpose(A - B * invR * transpose(B) * newP)*r - transpose(littlea) + newP * B * (invR) * littleb(:,index);
  
  rdot = rdot(:);
  
end

function zdot = getZdot(t, z, P, R, Q, As, Bs, xs, us, rs, littleA)
  global xdest N T trajectory littlea littleb invR;
  index = round((t/T)*(N-1)+1);
  A = As(:,:,index);
  B = Bs(:,:,index);
  newr1 = rs(index,1);
  newr2 = rs(index,2);
  Pval = P(index,:);
  newP = [Pval(1:2); Pval(3:4)];
  rval = [newr1, newr2];
  
  zdot = A * z + B * (-invR * transpose(B) * newP * z - invR * transpose(B) * transpose(rval) - invR * littleb(:,index));
end

function v = getV(R, As, Bs, P, zs, rs, xs, us, Q)
  global T N xdest littlea littleb invR;
  vs = [];
  us = transpose(us);
  for i=1:length(rs)
    B = Bs(:,:,i);
    Pval = P(i,:);
    newP = [Pval(1:2); Pval(3:4)];
    newu1 = us(1,i);
    newu2 = us(2,i);
    z = transpose(zs(i,:));
    rval = transpose(rs(i,:));
    vs(:,i) = invR * transpose(B) * newP * z - invR * transpose(B) * rval - invR * littleb(:,i);
  end
  v = transpose(vs);
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