// Update view based on file output every half second
$(document).ready(function() {
  const ROWS = 6;
  const COLS = 7;
  function parseText(data) {
    var squares = $('#board').children(".board-square");
    lines = data.split("\n");
    for (i = 0; i < lines.length; i++) {
      row = lines[i].split(" ");
      for (j = 0; j < row.length; j++) {
        rowMajor = COLS * i + j;
        if (row[j] === "R") {
          squares[rowMajor].children[0].className = "board-circle red-piece";
        } else if (row[j] === "Y") {
          squares[rowMajor].children[0].className = "board-circle yellow-piece";
        } else {
          squares[rowMajor].children[0].className = "board-circle";
        }
      }
    }
  }

  function updateText() {
    $.get('output/board.txt', function(data) {
      parseText(data);
    }, 'text');
    setTimeout(updateText, 500);
  }
  setTimeout(updateText, 10);
});
