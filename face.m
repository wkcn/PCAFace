file_dir = './orl_faces';
w = 92; h = 112;
n = 40;
data = zeros(n * 10, w * h);


k = 1;
for i = 1:n
	for j = 1:10
		name = [file_dir, '/s', int2str(i), '/', int2str(j),'.pgm'];
		pic = imread(name);
		d = reshape(pic, 1 , w * h);
		data(k, :) = d;
		k += 1;
	end
end

X = zeros(n * 7, w * h);
k = 1;
for i = 1:n
	for j = 1:7
		X(k, :) = data((i-1) * 10 + j,:);
		k += 1;
	end
end

avg = mean(X); % row vector;
W = X - avg;
[e, r] = size(W);
if r <= e
	Q = W' * W;
	[cv, D] = eig(Q);
else
	Q = W * W';
	[V, D] = eig(Q);
	cv = W' * V;
end

% sort feature vector by feature values
[D_sort, D_index] = sort(diag(D), 'descend');
pv = cv(:, D_index);

%recognize
%pv = eye(size(pv));
ev = pv(:, 1:30);
Y = W * ev;
right = 0;
for i = 1:n
	for j = 8:10
		p = data((i-1) * 10 + j, :)'; % column vector
		pa = p - avg'; % cvector
		y = ev' * pa;
		f = repmat(y', n * 7, 1); 
		d = Y - f;
		g = d .* d;
		su = sum(g, axis = 2);
		[dis, vr] = min(su);
		r = ceil(vr / 7);
		if i == r
			right += 1;
		end	
	end
end

right * 100 / (n * 3)
