<html>
% include('header.tpl', rand=rand)

<body>
	% include('navbar.tpl', query_word=query_word, w2v=w2v)
	<span id='imdb-key' class='hidden'>{{imdb_key}}</span>
	<div id='page'>
	<h2 id='movie-title'></h2>
	<div id="plot">
	<p>
	<img id='poster' src='' class="movie-poster" align="right">
	% for i in range(len(plot)):
		<span name="{{clean_words[i]}}" class="plot-word" style="background-color:rgb(255,{{255-int(weights[i][0])}},{{255-int(weights[i][0])}});">{{plot[i]}} </span>
	% end
	</p>
	</div>	
	
	</div>
	<script type="text/javascript" src="/static/plot_script.js?t={{rand}}"></script>
</body>
</html>