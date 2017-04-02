<html>
<head>
	<title>MovieQA Visualization</title>
	
    <link href="//fonts.googleapis.com/css?family=Raleway:400,300,600" rel="stylesheet" type="text/css">
    <link rel="stylesheet" href="/static/normalize.css">
  	<link rel="stylesheet" href="/static/skeleton.css">
  	<link rel="stylesheet" type="text/css" href="/static/style.css?t={{rand}}">
  	<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.0/jquery.min.js"></script>
</head>


<body>
	<div id='nav-bar'>
		<h1>
			Queried word: {{query_word}} 
			<div id='nav'>
			<span id = "tfidf">TF-IDF</span>   
			<span id = "w2v">Word2Vec</span>
			</div>
		</h1>
	</div>
	
	<span id='imdb-key' class='hidden'>{{imdb_key}}</span>
	<div id='page'>
	<h2 id='movie-title'></h2>
	<div id="plot">
	<p>
	<img id='poster' src='' align="right">
	% for i in range(len(plot)):
		<span name="{{clean_words[i]}}" class="plot-word" style="background-color:rgb(255,{{255-int(weights[i][0])}},{{255-int(weights[i][0])}});">{{plot[i]}} </span>
	% end
	</p>
	</div>	
	
	</div>
	<script type="text/javascript" src="/static/script.js"></script>
</body>
</html>