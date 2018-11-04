var calculateSlidesToShow = function () {
  var posterSize = 150;
  var marginSize = 5;
  var vw = $(document).width();
  var maxSlidesToShow = Math.floor(vw / (posterSize + marginSize)) - 1;
  if (maxSlidesToShow % 2 === 0) maxSlidesToShow--;   // Returns only odd numbers for proper center align.
  return maxSlidesToShow;
};

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

// TODO(Sejin): Make slick to adjust `slidesToShow` automatically with the window size.
// https://github.com/kenwheeler/slick/issues/1071#issuecomment-402330919
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
});
$(window).resize(function () {
  resetSlick('.movie-infos');
});
