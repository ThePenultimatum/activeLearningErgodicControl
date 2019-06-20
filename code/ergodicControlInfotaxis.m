clear;
global mapMat measurementLocation doorLocation doorLocation2 numUniqueVisited totalSpaces entropy epsilonEntropy T_t prior posterior iters valsAtPos visited;
mapMat = [];
totalSpaces = 25*25;
prior = ones(25,25) / totalSpaces;
posterior = ones(25,25) / totalSpaces;
measurementLocation = [round(rand()*24)+1 round(rand()*24)+1];
doorLocation = [round(rand() * 24)+1 round(rand() * 24)+1];
doorLocation2 = [round(rand() * 24)+1 round(rand() * 24)+1];
numUniqueVisited = 0;
epsilonEntropy = 0.1;

visited = zeros(25, 25);

T_t = measurementLocation;

%%%%
entropies = [];
entropy = entropyBoard();
entropies = [entropies, entropy];
valsAtPos = [];
iters = 0;











while entropy > 0.1 %iters < 100 %entropy > 0.1 %iters < 50 %entropy > 9 %< 5 %entropy > epsilonEntropy
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% take a measurement x
    
    x_i = takeMeasurement(); % 1 or 0 represents the value at the measurementLocation
    visited(measurementLocation(1), measurementLocation(2)) = 1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% update posterior p_x(theta)
    % this is the likelihood that the measurement x_i would turn out as the value we measured
    if (x_i == 0)
        likelihood_i = getLikelihoodNotDoor(measurementLocation(1), measurementLocation(2))
    else
        likelihood_i = getLikelihoodDoor(measurementLocation(1), measurementLocation(2));
    end
    
    posterior_i = (prior .* likelihood_i);
    posterior_i = posterior_i ./ (sum(sum(posterior_i)));
    posterior = posterior_i;
    prior = posterior_i
    
    entropy_i = entropyBoard();
    
    entropy = entropy_i;
    entropies = [entropies, entropy];
    
    % [0, 0], [0, 1], [1, 0], [-1, 0], [0, -1] are the 5 possible controls u_i
    
    u_i = getBestControls();%[0, 0];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% apply input u_i
    
    valsAtPos = [valsAtPos, x_i];
    
    prior = posterior;
    
    measurementLocation = measurementLocation + u_i;
    T_t = [T_t; measurementLocation];
    
    
    
    iters = iters + 1;
end

prior;
T_t;
entropies;
plot(T_t(:,1), T_t(:,2)); %%%%%%%%%%%%%%%%%%%%%% Plotting like x = rows, y = cols
title("Trajectory for Infotaxis");
xlim([-1 26]); 
ylim([-1 26]);
xlabel("x");
ylabel("y");
hold on;
plot(T_t(1,1), T_t(1,2),'.r','MarkerSize',40);
plot(doorLocation(1), doorLocation(2),'.b','MarkerSize',40);
plot(doorLocation2(1), doorLocation2(2),'.b','MarkerSize',40);
hold off;

%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS

function n = sumUniqueVisited()
    global prior visited;
    sumsofar = 0;
    for i=1:25
        for j=1:25
            if (visited(i,j) == 1)
                addon = prior(i,j);
                sumsofar = sumsofar + addon;
            end
        end
    end
    n = sumsofar;
end

function n = uniqueVisited()
    global visited;
    n = sum(sum(visited));
end

function us = getBestControls()
    global entropy measurementLocation prior totalSpaces numUniqueVisited;  
    
    %%%% below assumes first data part represents x while second represents
    %%%% y in plot
    entropyLeft1 = getEntLeft(1);
    entropyLeft0 = getEntLeft(0);
    
    entropyDown1 = getEntDown(1);
    entropyDown0 = getEntDown(0);
    
    entropyUp1 = getEntUp(1);
    entropyUp0 = getEntUp(0);
    
    entropyRight1 = getEntRight(1);
    entropyRight0 = getEntRight(0);
    
    entropyStay1 = getEntStay(1);
    entropyStay0 = getEntStay(0);
    
    probLeft1 = getPriorDoor(measurementLocation(1)-1, measurementLocation(2));
    probLeft0 = getPriorNotDoor(measurementLocation(1)-1, measurementLocation(2));
    
    probDown1 = getPriorDoor(measurementLocation(1), measurementLocation(2)-1);
    probDown0 = getPriorNotDoor(measurementLocation(1), measurementLocation(2)-1);
    
    probUp1 = getPriorDoor(measurementLocation(1), measurementLocation(2)+1);
    probUp0 = getPriorNotDoor(measurementLocation(1), measurementLocation(2)+1);
    
    probRight1 = getPriorDoor(measurementLocation(1)+1, measurementLocation(2));
    probRight0 = getPriorNotDoor(measurementLocation(1)+1, measurementLocation(2));
    
    probStay1 = getPriorDoor(measurementLocation(1), measurementLocation(2));
    probStay0 = getPriorNotDoor(measurementLocation(1), measurementLocation(2));
    
    expEntDiffLeft = probLeft1 * (entropy - entropyLeft1) + probLeft0 * (entropy - entropyLeft0);
    expEntDiffDown = probDown1 * (entropy - entropyDown1) + probDown0 * (entropy - entropyDown0);
    expEntDiffUp = probUp1 * (entropy - entropyUp1) + probUp0 * (entropy - entropyUp0);
    expEntDiffRight = probRight1 * (entropy - entropyRight1) + probRight0 * (entropy - entropyRight0);
    expEntDiffStay = probStay1 * (entropy - entropyStay1) + probStay0 * (entropy - entropyStay0);
    
    res = [expEntDiffLeft, expEntDiffDown, expEntDiffUp, expEntDiffRight, expEntDiffStay];
    [best, ind] = max(res);
    controlsPotential = [-1, 0; 0, -1; 0, 1; 1, 0; 0, 0];
    
    us = controlsPotential(ind,:);    
end

function h = getEntLeft(data)
    global entropy measurementLocation prior totalSpaces numUniqueVisited;
    if ((measurementLocation(1) - 1) < 1)
        h = 100000000000000;
    else
        tmp = prior;
        if (data == 1)
            tmpLikelihood = getLikelihoodDoor(measurementLocation(1)-1, measurementLocation(2));
        else
            tmpLikelihood = getLikelihoodNotDoor(measurementLocation(1)-1, measurementLocation(2));
        end
        tmpPost = (tmp .* tmpLikelihood);
        tmpPost = tmpPost ./ (sum(sum(tmpPost)));
        h = entropyOfBoard(tmpPost);
    end
end

function h = getEntDown(data)
    global entropy measurementLocation prior totalSpaces numUniqueVisited;
    if ((measurementLocation(2) - 1) < 1)
        h = 100000000000000;
    else
        tmp = prior;
        if (data == 1)
            tmpLikelihood = getLikelihoodDoor(measurementLocation(1), measurementLocation(2)-1);
        else
            tmpLikelihood = getLikelihoodNotDoor(measurementLocation(1), measurementLocation(2)-1);
        end
        tmpPost = (tmp .* tmpLikelihood);
        tmpPost = tmpPost ./ (sum(sum(tmpPost)));
        h = entropyOfBoard(tmpPost);
    end
end

function h = getEntUp(data)
    global entropy measurementLocation prior totalSpaces numUniqueVisited;
    if ((measurementLocation(2) + 1) > 25)
        h = 100000000000000;
    else
        tmp = prior;
        if (data == 1)
            tmpLikelihood = getLikelihoodDoor(measurementLocation(1), measurementLocation(2)+1);
        else
            tmpLikelihood = getLikelihoodNotDoor(measurementLocation(1), measurementLocation(2)+1);
        end
        tmpPost = (tmp .* tmpLikelihood);
        tmpPost = tmpPost ./ (sum(sum(tmpPost)));
        h = entropyOfBoard(tmpPost);
    end
end

function h = getEntRight(data)
    global entropy measurementLocation prior totalSpaces numUniqueVisited;
    if ((measurementLocation(1) + 1) > 25)
        h = 100000000000000;
    else
        tmp = prior;
        if (data == 1)
            tmpLikelihood = getLikelihoodDoor(measurementLocation(1)+1, measurementLocation(2));
        else
            tmpLikelihood = getLikelihoodNotDoor(measurementLocation(1)+1, measurementLocation(2));
        end
        tmpPost = (tmp .* tmpLikelihood);
        tmpPost = tmpPost ./ (sum(sum(tmpPost)));
        h = entropyOfBoard(tmpPost);
    end
end

function h = getEntStay(data)
    global entropy measurementLocation prior totalSpaces numUniqueVisited;
    h = entropy;
end

function s = entropyBoard()
  global posterior;
  total = 0;
  for i=1:25
      for j=1:25
          ijval = posterior(i, j);
          if (ijval == 0)
              total = total;
          else
              total = total + ijval * log2(ijval);
          end
      end
  end
  s = -1*total;
end

function s = entropyOfBoard(b)
  global posterior;
  total = 0;
  for i=1:25
      for j=1:25
          ijval = b(i, j);
          if (ijval == 0)
              total = total;
          else
              total = total + ijval * log2(ijval);
          end
      end
  end
  s = -1*total;
end

function x = takeMeasurement()
  global measurementLocation doorLocation doorLocation2;
  if ((measurementLocation(1) == doorLocation(1)) && (measurementLocation(2) == doorLocation(2))) || ((measurementLocation(1) == doorLocation2(1)) && (measurementLocation(2) == doorLocation2(2)))
      x = 1;
  else
      r = rand();
      l = getLikelihoodDoorMeasurement(measurementLocation(1), measurementLocation(2))
      if (r < l)
          x = 1
      else
          x = 0
      end
  end
end

function l = getLikelihoodDoorMeasurement(r,c)
    global doorLocation doorLocation2;
    rowdiff = abs(r - doorLocation(1));
    coldiff = abs(c - doorLocation(2));
    rowdiff2 = abs(r - doorLocation2(1));
    coldiff2 = abs(c - doorLocation2(2));    
    diff = coldiff - rowdiff;
    diff2 = coldiff2 - rowdiff2;
    if ((diff <= 0) && (rowdiff < 4))
        if (rowdiff == 3)
            l = 1/4;
        else
            if (rowdiff == 2)
                l = 1/3;
            else
                if (rowdiff == 1)
                    l = 1/2;
                else
                    if (rowdiff == 0)
                        l = 1;
                    end
                end
            end
        end
    else
        if ((diff2 <= 0) && (rowdiff2 < 4))
            if (rowdiff2 == 3)
                l = 1/4;
            else
                if (rowdiff2 == 2)
                    l = 1/3;
                else
                    if (rowdiff2 == 1)
                        l = 1/2;
                    else
                        if (rowdiff2 == 0)
                            l = 1;
                        end
                    end
                end
            end
        else
            l = 1/100;
        end
    end
end

function pn = getPriorNotDoor(i, j) %%%%%%%%%%% MAKE SURE THAT THIS IS ONLY USED WHEN CHECKING FOR NON_VISITED SPACE
  pn = 1-(getPriorDoor(i, j));
end

function pr = getPriorDoor(i, j) %%%%%%%%%%% MAKE SURE THAT THIS IS ONLY USED WHEN CHECKING FOR NON_VISITED SPACE
  global prior;
  if ((i > 0) && (i < 26) && (j > 0) && (j < 26))
      pr = prior(i, j);%1/(totalSpaces - numUniqueVisited);
  else
      pr = 0;
  end
end

function l = getLikelihoodDoor(r, c)
    tmp = ones(25, 25)./100;
    tmp(r,c) = 1;
    for i=0:3
        if (((r-i) > 0) && ((r+i)<26))
            for j=0:3
                if (((c-j) > 0) && ((c+j)<26))
                    if (i == 3)
                        tmp(r-i, c+j) = 1/4;
                        tmp(r-i, c-j) = 1/4;
                        tmp(r+i, c-j) = 1/4;
                        tmp(r+i, c+j) = 1/4;
                    end
                    if ((i == 2) && ((j == 0) || (j == 1) || (j == 2)))
                        tmp(r-i, c+j) = 1/3;
                        tmp(r-i, c-j) = 1/3;
                        tmp(r+i, c-j) = 1/3;
                        tmp(r+i, c+j) = 1/3;
                    end
                    if ((i == 1) && ((j == 0) || (j == 1)))
                        tmp(r-i, c+j) = 1/2;
                        tmp(r-i, c-j) = 1/2;
                        tmp(r+i, c-j) = 1/2;
                        tmp(r+i, c+j) = 1/2;
                    end
                    if ((i == 0) && (j == 0))
                        tmp(r,c) = 1;
                    end
                end
            end
        end
    end
    l = tmp;
end

function l = getLikelihoodNotDoor(r, c)
    tmp = ones(25, 25).*99./100;
    tmp(r,c) = 0;
    for i=0:3
        if (((r-i) > 0) && ((r+i)<26))
            for j=0:3
                if (((c-j) > 0) && ((c+j)<26))
                    if (i == 3)
                        tmp(r-i, c+j) = 3/4;
                        tmp(r-i, c-j) = 3/4;
                        tmp(r+i, c-j) = 3/4;
                        tmp(r+i, c+j) = 3/4;
                    end
                    if ((i == 2) && ((j == 0) || (j == 1) || (j == 2)))
                        tmp(r-i, c+j) = 2/3;
                        tmp(r-i, c-j) = 2/3;
                        tmp(r+i, c-j) = 2/3;
                        tmp(r+i, c+j) = 2/3;
                    end
                    if ((i == 1) && ((j == 0) || (j == 1)))
                        tmp(r-i, c+j) = 1/2;
                        tmp(r-i, c-j) = 1/2;
                        tmp(r+i, c-j) = 1/2;
                        tmp(r+i, c+j) = 1/2;
                    end
                    if ((i == 0) && (j == 0))
                        tmp(r,c) = 0;
                    end
                end
            end
        end
    end
    l = tmp;
end