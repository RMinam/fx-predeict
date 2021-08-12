<html>
  <body>
    <h1>FX診断</h1>
    <form method="post" action="/start" enctype="multipart/form-data">
      csv1を入れてください(今は機能していない): <input type="file" name="input_csv1"><br>
      csv2を入れてください(今は機能していない): <input type="file" name="input_csv2"><br>
      csv3を入れてください(今は機能していない): <input type="file" name="input_csv3"><br>
      csv4を入れてください(今は機能していない): <input type="file" name="input_csv4"><br>
      csv5を入れてください(今は機能していない): <input type="file" name="input_csv5"><br>
      期間を入れてください: <input type="number" name="input_date"><br>
      max_depthを入れてください: <input type="number" name="input_max_depth"><br>
      割合を入れてください: <input type="range" name="input_ratio" min="1" max"100"><br>
      <input type="submit" value="送信">
    </form>
    <ul>
      <li>結果:  {{!output_text}}</li>
    </ul>
  </body>
</html>
