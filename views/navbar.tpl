<div id='nav-bar'>
	<h1>
		% if query_word:
			Queried word: 
			<span class='selected'>
			{{query_word}} 
			</span>
		% end
		<div id='nav'>
		<a href="/tfidf/">
		<span id = "tfidf"
		% if not w2v:
			class = "selected"
		% end
		>TF-IDF</span>   
		</a>
		<a href="/w2v/">
		<span id = "w2v" class="white 
		% if w2v:
			selected
		% end
		 ">
		Word2Vec</span>
		</a>
		</div>
	</h1>
</div>