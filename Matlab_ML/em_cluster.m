function [R, Mu] = em_cluster (D,K,iterations)

    %initialize first cluster assumption with kmeans algorithm
    [t,Mu] = kmeans_cluster(D,K);
    for k = (1:K)
        cluster = D(t==k,:);
        Sig{k} = cov(cluster);
        pai(k) = size(cluster,1)/size(D,1);
    end
    
    for i = 1:iterations
        
        R = e_step (D, K, pai, Mu, Sig);
        [Mu, Sig, pai] = m_step (D, R);
        
         for k = 1:K
            logllikelihood(:,k) = pai(k).*prod((1./sqrt(2*pi()*diag(Sig{k})')) .* exp((-(1/2)./diag(Sig{k})') .* ((D-Mu(k,:)).^2)),2);
         end
         LogLikelihood(i) = sum(log(sum(logllikelihood,2)));
    end
    
    figure
    plot([1:iterations],LogLikelihood)
    title('Log likelihood Progression')
    xlabel('Iteration')
    ylabel('Log likelihood')
    
end