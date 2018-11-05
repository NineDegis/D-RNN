movieID = 0;
currentSlide = 0;

/* ===== functions for slick ===== */

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


/* ===== functions for ajax call ===== */

var refreshReviews = function (reviewList) {
  var reviewsElem = $(`.movie-detail:eq(${currentSlide}) .reviews`);
  var numComments = reviewList.length;
  if (numComments === 0) {
    reviewsElem.html("<p class='no-data'> There is no reviews for this movie :( </p>");
    return;
  }
  reviewsElem.html("");
  for (var i = 0; i < numComments; i++) {
    reviewsElem.append(
      "<div class='review'>" +
      "<p class='datetime'>" +
      reviewList[i].review_datetime +
      "</p>" +
      "<p class='comment'>" +
      reviewList[i].comment +
      "</p></div>"
    );
  }
};

var fetchReviews = function () {
  $.ajax({
    url: '/reviews/read?id=' + movieID,
    success: function (reviewList) {
      refreshReviews(reviewList);
    }
  });
};

var insertReview = function (reviewComment) {
  $.ajax({
    url: '/reviews/write?id=' + movieID,
    type: 'POST',
    data: {'comment': reviewComment},
    success: function (reviewList) {
      refreshReviews(reviewList);
    },
  });
};


/* ===== main events ===== */

$(document).ready(function () {
  var movieDetailsElem = $('.movie-details');

  movieDetailsElem.slick({
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
  currentSlide = 0;
  movieID = $('.movie-detail')[0].id;
  fetchReviews(0);

  movieDetailsElem.on('afterChange', function (event, slick, currentSlide, nextSlide) {
    window.currentSlide = currentSlide; //????
    movieID = $(`.movie-detail:eq(${currentSlide})`)[0].id;
    fetchReviews();
  });
  $('.review-input button').click(function () {
    var reviewComment = $(`.review-textarea:eq(${currentSlide})`)[0].value;
    insertReview(reviewComment);
  });
});

$(window).resize(function () {
  resetSlick('.movie-infos');
});
