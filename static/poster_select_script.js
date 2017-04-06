"use strict";

var get_poster = function(){
	var imdb_key = $( this ).attr('id');
	$.ajax({
        url: 'http://www.omdbapi.com/?i=' + imdb_key,
        success: show_poster,
        context: this
    });
};

var show_poster = function(res){
	console.log(res);
	$(this).attr("src", res.Poster);
};

var go_to_movie = function(){
	var imdb_key = $( this ).attr('id')
	window.location.href = imdb_key + '/';
};

var get_poster_local = function(){
	$(this).attr("src", '/poster/'+ $(this).attr('id')+'.jpg');
};

$(".movie-poster").each(get_poster_local);
$(".movie-poster").click(go_to_movie);