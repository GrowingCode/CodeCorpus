package com.zzk;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import com.itextpdf.text.BaseColor;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Font;
import com.itextpdf.text.PageSize;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.exceptions.IllegalPdfSyntaxException;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfContentByte;
import com.itextpdf.text.pdf.PdfPageEventHelper;
import com.itextpdf.text.pdf.PdfTemplate;
import com.itextpdf.text.pdf.PdfWriter;

public class AddPageNumber extends PdfPageEventHelper {

	public PdfTemplate pdfTemplate;// 声明模板对象
	public BaseFont baseFont;// 声明基础字体对象

	public static void main(String[] args) {
		Document document = new Document(PageSize.A4);// 创建A4纸张大小的PDF文档对象
		try {
			PdfWriter writer = PdfWriter.getInstance(document,new FileOutputStream("C:\\添加页码.pdf"));// 关联文档对象与输出流 
			writer.setPageEvent(new AddPageNumber());// 添加页面事件监听器
			document.open();// 打开文档
			BaseFont bf = BaseFont.createFont("STSong-Light", "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基础字体
			Paragraph pargraph = new Paragraph("添加页码和总页数。", new Font(bf, 15,
					Font.BOLD, BaseColor.BLUE));// 创建段落对象并指定中文
			document.add(pargraph);// 向文档中添加段落
			document.newPage();// 创建新页
			document.add(pargraph);// 添加段落
			document.newPage();// 创建新页
			document.add(pargraph);// 添加段落
			document.close();// 关闭文档
		} catch (IllegalPdfSyntaxException de) {
			de.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public void onOpenDocument(PdfWriter writer, Document document) {// 打开文档时执行该方法
			pdfTemplate = writer.getDirectContent().createTemplate(180, 180);// 创建模板对象
			try {
				baseFont = BaseFont.createFont("STSong-Light", "UniGB-UCS2-H",
						BaseFont.NOT_EMBEDDED);// 创建基础字体
			} catch (DocumentException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
	}

	public void onEndPage(PdfWriter writer, Document document) {// 结束页时执行该方法
		PdfContentByte cb = writer.getDirectContent();// 创建内容对象
		cb.saveState();// 保存状态
		cb.beginText();// 文本开始标记
		cb.setFontAndSize(baseFont, 12);// 设置字体和字号
		cb.setTextMatrix(260, 800);// 设置文本显示位置
		String page = "第" + writer.getPageNumber() + "页/共";// 定义表示页码信息的变量
		cb.showText(page);// 显示文本
		cb.endText();// 文本结束标记
		cb.addTemplate(pdfTemplate, 305, 800);// 添加模板对象
		cb.stroke();// 确认并保存操作
		cb.restoreState();// 恢复标记
		cb.closePath();// 关闭内容通道
	}

	public void onCloseDocument(PdfWriter writer, Document document) {// 关闭文档时执行该方法
		pdfTemplate.beginText();// 模板文本开始标记
		pdfTemplate.setFontAndSize(baseFont, 12);// 设置模板字体和字号
		int totalPages = writer.getPageNumber() - 1;// 获得总页数
		pdfTemplate.showText(totalPages + "页");// 显示总页数信息
		pdfTemplate.endText();// 模板文本结束标记
		pdfTemplate.closePath();// 关闭模板通道
	}
}
