/* One class: adult butterfly or moth. Rectangles use relative coordinates. */
const BBOX_POLICY = Object.freeze({ iou: 0.50, coverage: 0.85, precision: 0.55 });
const BBOX_EPS = 1e-6;

function parseRects(value) {
  if (typeof value === "string") value = JSON.parse(value);
  if (!Array.isArray(value)) throw new Error("Ожидается список прямоугольников.");
  return value.map(shape => {
    let x1, y1, x2, y2;
    if (shape && shape.type === "rectangle" && shape.data?.p1 && shape.data?.p2) {
      const {p1, p2} = shape.data;
      if (![p1.x,p1.y,p2.x,p2.y].every(Number.isFinite)) throw new Error("Некорректные координаты.");
      x1 = Math.min(p1.x,p2.x); y1 = Math.min(p1.y,p2.y);
      x2 = Math.max(p1.x,p2.x); y2 = Math.max(p1.y,p2.y);
    } else if (shape && !shape.type && [shape.left,shape.top,shape.width,shape.height].every(Number.isFinite)) {
      ({left:x1, top:y1} = shape);
      x2 = x1 + shape.width; y2 = y1 + shape.height;
    } else {
      throw new Error("Используйте прямоугольные рамки.");
    }
    if (Math.min(x1,y1) < -BBOX_EPS || Math.max(x2,y2) > 1+BBOX_EPS || x2<=x1 || y2<=y1) {
      throw new Error("Рамка должна иметь площадь и находиться внутри кадра.");
    }
    return {x1:Math.max(0,x1), y1:Math.max(0,y1), x2:Math.min(1,x2), y2:Math.min(1,y2)};
  });
}

function measures(mark, reference) {
  const area = b => (b.x2-b.x1)*(b.y2-b.y1);
  const common = Math.max(0,Math.min(mark.x2,reference.x2)-Math.max(mark.x1,reference.x1)) *
                 Math.max(0,Math.min(mark.y2,reference.y2)-Math.max(mark.y1,reference.y1));
  return {
    iou: common/(area(mark)+area(reference)-common),
    coverage: common/area(reference),
    precision: common/area(mark),
  };
}

function gradeBoxes(submitted, reference, policy=BBOX_POLICY) {
  let expected, marks;
  try { expected = parseRects(reference); }
  catch { return {ok:false,code:"invalid_reference",message:"Эталон задания повреждён. Сообщите заказчику."}; }
  try { marks = parseRects(submitted); }
  catch (e) { return {ok:false,code:"invalid_response",message:e.message}; }
  if (marks.length !== expected.length) {
    return {ok:false,code:"count",message:"Проверьте число рамок. На каждую видимую взрослую бабочку нужна отдельная рамка."};
  }

  const edges = marks.map(mark => expected.map((target,i) => ({i,...measures(mark,target)}))
    .filter(m => m.iou+BBOX_EPS>=policy.iou && m.coverage+BBOX_EPS>=policy.coverage &&
      m.precision+BBOX_EPS>=policy.precision).sort((a,b)=>b.iou-a.iou));
  const owner = new Array(expected.length).fill(-1);
  function match(mark,seen) {
    for (const {i} of edges[mark]) {
      if (seen.has(i)) continue;
      seen.add(i);
      if (owner[i]===-1 || match(owner[i],seen)) { owner[i]=mark; return true; }
    }
    return false;
  }
  if (!marks.every((_,i)=>match(i,new Set()))) {
    return {ok:false,code:"geometry",message:"Проверьте границы. Включите видимые крылья, тело, усики и лапки. Уберите лишний фон."};
  }
  return {ok:true,code:"pass",message:"Верно. Можно переходить к следующему кадру."};
}

function gradeAnswer(status, result, reference) {
  if (status==="load_error") return {ok:false,code:"load_error",message:"Изображение не загрузилось. Пропустите задание или сообщите заказчику."};
  if (status!=="boxes" && status!=="none") return {ok:false,code:"status",message:"Нарисуйте рамки или явно отметьте, что взрослых бабочек нет."};
  let marks;
  try { marks=parseRects(result); }
  catch (e) { return {ok:false,code:"invalid_response",message:e.message}; }
  if ((status==="none") !== (marks.length===0)) {
    return {ok:false,code:"status",message:"Ответ «Нет взрослых бабочек» должен быть без рамок."};
  }
  return gradeBoxes(result,reference);
}

if (typeof module !== "undefined" && module.exports) {
  module.exports={gradeBoxes,gradeAnswer,parseRects,measures,BBOX_POLICY};
}

function mountBoxTraining(section, reference, saved, commit) {
  const image = section.querySelector(".bbox-photo");
  const layer = section.querySelector(".bbox-layer");
  const feedback = section.querySelector(".bbox-feedback");
  let shapes = Array.isArray(saved.result) ? saved.result : [];
  let status = saved.status || "unset";
  let pointer = null;
  let start = null;
  let loaded = image.complete && image.naturalWidth > 0;
  function point(event) {
    const rect = layer.getBoundingClientRect();
    const unit = value => Math.min(1,Math.max(0,value));
    return {x:unit((event.clientX-rect.left)/rect.width),y:unit((event.clientY-rect.top)/rect.height)};
  }
  function draw(draft) {
    const current = draft ? shapes.concat(draft) : shapes;
    const nodes = current.map(shape => {
      const {p1,p2} = shape.data;
      const node = document.createElementNS("http://www.w3.org/2000/svg","rect");
      node.setAttribute("x",Math.min(p1.x,p2.x));
      node.setAttribute("y",Math.min(p1.y,p2.y));
      node.setAttribute("width",Math.abs(p2.x-p1.x));
      node.setAttribute("height",Math.abs(p2.y-p1.y));
      return node;
    });
    layer.replaceChildren(...nodes);
  }
  function change() {
    draw();
    commit({status,result:shapes,passed:loaded && gradeAnswer(status,shapes,reference).ok});
  }
  function check() {
    const verdict = !loaded && status!=="load_error"
      ? {ok:false,message:"Дождитесь загрузки изображения или отметьте ошибку загрузки."}
      : gradeAnswer(status,shapes,reference);
    feedback.textContent = verdict.message;
    commit({status,result:shapes,passed:verdict.ok});
    return verdict;
  }
  layer.addEventListener("pointerdown",event => {
    if (!loaded || pointer!==null || event.button!==0 || shapes.length>=100) return;
    pointer = event.pointerId;
    start = point(event);
    layer.setPointerCapture(pointer);
  });
  layer.addEventListener("pointermove",event => {
    if (event.pointerId===pointer) draw({type:"rectangle",data:{p1:start,p2:point(event)}});
  });
  layer.addEventListener("pointerup",event => {
    if (event.pointerId!==pointer) return;
    const end = point(event);
    const bounds = layer.getBoundingClientRect();
    if (Math.abs(end.x-start.x)*bounds.width>=3 && Math.abs(end.y-start.y)*bounds.height>=3) {
      shapes = shapes.concat({type:"rectangle",data:{p1:start,p2:end}});
      status = "boxes";
      feedback.textContent = "";
    }
    pointer = null;
    start = null;
    change();
  });
  layer.addEventListener("pointercancel",event => {
    if (event.pointerId!==pointer) return;
    pointer = null;
    start = null;
    draw();
  });
  image.addEventListener("load",()=>{loaded=true; if (status!=="unset") change();});
  image.addEventListener("error",()=>{loaded=false;feedback.textContent="Изображение не загрузилось.";});
  section.querySelector('[data-bbox="undo"]').addEventListener("click",()=>{
    shapes=shapes.slice(0,-1);
    status=shapes.length?"boxes":"unset";
    feedback.textContent="";
    change();
  });
  for (const selection of ["none","load_error"]) {
    section.querySelector('[data-bbox="'+selection+'"]').addEventListener("click",()=>{
      shapes=[];
      status=selection;
      feedback.textContent=selection==="none"?"Отмечено отсутствие взрослых бабочек.":"Отмечена ошибка загрузки.";
      change();
    });
  }
  section.querySelector('[data-bbox="check"]').addEventListener("click",check);
  draw();
  return {check,isLoaded:()=>loaded};
}

if (typeof TolokaHandlebarsTask !== "undefined") {
  function BoxTrainingTask(options) {
    TolokaHandlebarsTask.call(this,options);
  }
  BoxTrainingTask.prototype = Object.create(TolokaHandlebarsTask.prototype);
  Object.assign(BoxTrainingTask.prototype, {
    onRender: function () {
      const root = this.getDOMElement();
      const section = root.matches(".bbox-lesson") ? root : root.querySelector(".bbox-lesson");
      const input = this.getTask().input_values;
      const output = this.getSolution()?.output_values || {};
      this.bboxLesson = mountBoxTraining(section,input.reference,output,
        values=>this.setSolutionOutputValues(values));
    },
    validate: function (solution) {
      const output = (solution || this.getSolution())?.output_values || {};
      const verdict = gradeAnswer(output.status,output.result,this.getTask().input_values.reference);
      if (this.bboxLesson && !this.bboxLesson.isLoaded() && output.status!=="load_error") {
        const waiting = this.bboxLesson.check();
        return {result:{code:"not_loaded",message:waiting.message}};
      }
      if (!verdict.ok) {
        this.bboxLesson?.check();
        return {result:{code:verdict.code,message:verdict.message}};
      }
      return TolokaHandlebarsTask.prototype.validate.call(this,solution);
    }
  });
  exports.Task = BoxTrainingTask;
}
