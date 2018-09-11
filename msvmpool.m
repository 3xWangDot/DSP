% inputs
% seq: cell array with each cell one sequence, n x d, where d is the
% feature dim and n is the num frames
% Z a noise matrix of size num_noise x d
% num_subspaces : num of columns in subspace matrix which is returned.
% thresh: for the ordering constraints. 
% num_iter: for the manopt optimizer
% verbose: in case you want to show the ordered relations before and after
% the optimization.
% output: grp_desc: a cell array length same as the input seq, each cell
% has the subspace of size dxnum_subspaces. 
% to classify the sequence use a grassmannian kenrel in mypdist2. that is,
% mypdist2(X1, X2, 'grass'); and then use a kernel svm with -t 4.
%
% for questions, contact anoop.cherian@gmail.com
%
function [msvmp_fwd, msvmp_bwd] = msvmpool(input, Z, num_subspaces, thresh, num_iter, verbose)
seq{1}=input;
[msvmp_fwd, msvmp_bwd] = deal(cell(length(seq),1));
mysmooth = @(X) normr(X);% normr(cumsum(X,1));
nonlinear =@(X) sign(X).*sqrt(abs(X));

C=1; lambda=1; % change the parameter here.
for tt=1:length(seq)   
    xx = seq{tt}; xx=double(xx); 
    % if features are column vectors, tranpose them. we need nxd matrices.
    if size(xx,1)> size(xx,2)
        xx=xx';
    end
    
	n = size(xx,1); 
    % if you don't have sufficient frames, just put zeros.
    if n<num_subspaces
        dd=num_subspaces; %max(num_subspaces, );
        [d,n]=size(xx); xx(:,n+1:dd) = zeros(d, dd-n);
    end
		
    % compute multiple-decision boundaries for every sequence.
	msvmp_fwd{tt} = compute_seq_subspace((mysmooth(xx)), Z, num_subspaces, thresh, lambda, C, num_iter, verbose);  
    msvmp_bwd{tt} = compute_seq_subspace((mysmooth(xx(end:-1:1, :))), Z, num_subspaces, thresh, lambda, C, num_iter, verbose);  
end
end

function U = compute_seq_subspace(X, Z, p, thresh, lambda, C, num_iter, verbose)
    [n,d] = size(X); nz = size(Z,1); nxz = n+nz;
    L = [ones(n,1); -ones(nz,1)];
    XZ = [X; Z];
        
    idx = find(~tril(ones(n)));
    
    get_ux = @(u) sum((X*u).^2,2); % get number of ordering constraint viloations.
    get_acc = @(u) nnz(max(bsxfun(@times, XZ*u, L), [], 2)>0)/nxz; % get classification accuracy.
    
    manifold.U = stiefelfactory(d, p);              
    manifold.E = euclideanfactory(1,n*(n-1)/2);    
    problem.M = productmanifold(manifold);
    problem.cost = @objective;
    problem.egrad = @grad;
        
    %checkgradient(problem); return;
    
    U = initialize_U();
    xi = (thresh/1000)*ones(1,n*(n-1)/2)/1; % xi is the slack. initialize it.
            
    if verbose == 1
        % plot the temporal ordering before and after U is learned.   
        ux = get_ux(U);
        [vi, vj] = get_violations(U,xi); violate_init = length(vi);
        figure(1); subplot(1,2,1); plot(ux); title('temporal order initially'); %figure(3); plot(xi); title('xi');
        %fprintf('before: num_violations=%d\n', length(vi));
    end
        
    M_init.U = U; M_init.E = xi;  
    
    opts.maxiter = num_iter;
    opts.verbosity = 0;
    out = conjugategradient(problem, M_init, opts);
                
    U = out.U; xi = out.E;      
    
    if verbose == 1
        ux = get_ux(U);
        [vi, vj] = get_violations(U, xi); violate_final = length(vi);
        figure(1); subplot(1,2,2); plot(ux); title('temporal order after learning U');%figure(4); plot(xi); title('xi');
        fprintf('initial constraint violations = %d final: num_violations=%d: pos/neg acc=%f\n', violate_init, violate_final, get_acc(U)); 
        pause(1);
    end
          
    function obj = objective(M)    
        U = M.U; xi = M.E;                 
        sqUX = get_ux(U);
        sqxi = xi'.^2;

        % new obj.
        obj = sum(max(0, 1-max(bsxfun(@times, XZ*U, L),[], 2))); % XZ is nxp, L is n. 
        obj = obj + C*sum(sqxi);
        
        % ranking constraints.
        vv = bsxfun(@minus, sqUX, sqUX'); % (i,j)-th element in vv is ||UX_i||^2 - ||UX_j||^2.
        vv = vv(idx);        % takes the upper triangle of vv. 
        obj = obj + 0.5*lambda*sum(max(0,vv + thresh - sqxi));    
    end

    function gM = grad(M)
        U = M.U; xi = M.E; UX = X*U; UXZ = XZ*U;
        
        % gradient of the first term = -[0, yx_k, 0], where k is the column
        % corresponding to max(L\odot U*XZ).
        [mm, midx] = max(bsxfun(@times, UXZ, L),[], 2);
        gU = zeros(d,p);
        for t=1:nxz
            if mm(t) < 1
                ii = midx(t);
                gU(:,ii) = gU(:,ii) - L(t)*XZ(t,:)';
            end
        end
        
        gxi = 2*C*xi;
        [vi,vj, vxi] = get_violations(U, xi);
        if ~isempty(vi)
            uvi = unique(vi); uvj=unique(vj); 
            if length(uvi)>1, hvi = hist(vi, uvi)';  else hvi = length(vi); end
            if length(uvj)>1, hvj = hist(vj, uvj)';  else hvj = length(vj); end            
            
            Xi = bsxfun(@times, X(uvi,:), hvi);
            Xj = bsxfun(@times, X(uvj,:), hvj);            
            UXi = UX(uvi, :); UXj = UX(uvj, :);
            
            gU = gU + lambda*(Xi'*UXi - Xj'*UXj);            
            gxi = gxi - lambda*vxi;
        end        
        gM.U = gU; gM.E = gxi;
    end

    % use U and compute all violations. 
    function [vi,vj, vxi] = get_violations(U, xi)       
       trsqUX = get_ux(U);       
       vv = bsxfun(@minus, trsqUX, trsqUX') + thresh;       
       vv = vv(idx)-xi'.^2;
       ii = idx(vv>0);
       if any(ii)
           [vi,vj] = ind2sub([n,n], ii);
       else
           [vi,vj] = deal([]);
       end
       vxi = xi; vxi(vv<0) = 0;
    end
    
    function U = initialize_U() 
        %U = pca(X, 'NumComponents', p, 'Economy', true, 'Centered', false);                
        [U,~,~] = svds(X',p);
    end
end
