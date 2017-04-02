"use strict";

var imdb_key = $('#imdb-key').html()

function get_poster(){
	$.ajax({
        url: 'http://www.omdbapi.com/?i=' + imdb_key,
        success: show_poster
    });
}

var show_poster = function(res){
	console.log(res);
	$("#poster").attr("src", res.Poster);
	$("#movie-title").html(res.Title)
}

function select_word(){
	window.location.href = $( this ).attr('name');
}

$(".plot-word").click(select_word);

get_poster()