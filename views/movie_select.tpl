<html>
% include('header.tpl', rand=rand)

<body>
	% include('navbar.tpl', query_word=None, w2v=w2v)
	
	<div id='page'>
	<h2 id='movie-title'></h2>
	<div id="plot">
	<p>
	
	% for i in range(rows):
		<div class="row">
		% for j in range(min(len(imdb_keys)-i*6, 6)):
			<img id="{{imdb_keys[i*6+j]}}" class="movie-poster two columns">
		%end
		</div><br/>
	% end
	</p>
	</div>	
	
	</div>
	<script type="text/javascript" src="/static/poster_select_script.js?t={{rand}}"></script>
</body>
</html>