/* functions for slick */

var calculateSlidesToShow = function () {
  var posterSize = 150;
  var marginSize = 5;
  var vw = $(document).width();
  var maxSlidesToShow = Math.floor(vw / (posterSize + marginSize)) - 1;
  if (maxSlidesToShow % 2 === 0) maxSlidesToShow--;   // Returns only odd numbers for proper center align.
  return maxSlidesToShow;
};

// TODO(sejin): Make this function more flexible
var resetSlick = function (className) {
  $(className).slick('unslick');
  $(className).slick({
    slidesToShow: calculateSlidesToShow(),
    slidesToScroll: 1,
    asNavFor: '.movie-details',
    dots: false,
    centerMode: true,
    focusOnSelect: true,
  });
  $('.slick-prev').html('<');
  $('.slick-next').html('>');
};

$(document).ready(function () {
  $('.movie-details').slick({
    slidesToShow: 1,
    slidesToScroll: 1,
    arrows: false,
    fade: true,
    asNavFor: '.movie-infos',
  });
  $('.movie-infos').slick({
    slidesToShow: calculateSlidesToShow(),
    slidesToScroll: 1,
    asNavFor: '.movie-details',
    dots: false,
    centerMode: true,
    focusOnSelect: true,
  });
  $('.slick-prev').html('<');
  $('.slick-next').html('>');
  const movieID = $('.movie-detail')[0].id;
  fetchReviews(movieID, 0);
});
$(window).resize(function () {
  resetSlick('.movie-infos');
});



/* functions for ajax call */
var fetchReviews = function(movieID, currentSlide) {
  $.ajax({
    url: '/reviews?id=' + movieID,
    success: function (reviewList) {
      var reviewsElem = $(`.movie-detail:eq(${currentSlide}) .reviews`);
      console.log(reviewsElem);
      var numComments = reviewList.length;
      if (numComments === 0) {
        reviewsElem.html("<p class='no-data'> There is no reviews for this movie :( </p>");
        return;
      }
      reviewsElem.html("");
      for (var i = 0; i < numComments; i++) {
        reviewsElem.append(
          "<div>" +
          "<p class='datetime'>" +
          reviewList[i].review_datetime +
          "</p>" +
          "<p class='comment'>" +
          reviewList[i].comment +
          "</p></div>"
        );
      }
    }
  });
};

$('.movie-details').on('afterChange', function(event, slick, currentSlide, nextSlide) {
  const movieID = $('.movie-detail')[currentSlide].id;
  fetchReviews(movieID, currentSlide);
});
