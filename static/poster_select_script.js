"use strict";

var get_poster = function(){
	var imdb_key = $( this ).attr('id')
	$.ajax({
        url: 'http://www.omdbapi.com/?i=' + imdb_key,
        success: show_poster,
        context: this
    });
}

var show_poster = function(res){
	console.log(res);
	$(this).attr("src", res.Poster);
}

var go_to_movie = function(){
	var imdb_key = $( this ).attr('id')
	window.location.href = imdb_key + '/';
}

$(".movie-poster").each(get_poster);
$(".movie-poster").click(go_to_movie);